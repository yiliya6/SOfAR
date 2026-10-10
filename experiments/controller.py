"""Dynamic bicycle, MIQP side choice, biased MPPI, and event-triggered shield.

All methods share the same model, samples per cycle, geometry, and fallback.
This is a new experimental controller, not the paper's acados implementation.
"""

from dataclasses import dataclass
from time import perf_counter

import gurobipy as gp
import numpy as np
import osqp
from numba import njit
from scipy import sparse

from experiments.track import Track

DT = 0.1
H = 20
K = 128


@njit(cache=True)
def geometry(s: float, line: np.ndarray, length: float) -> np.ndarray:
    arc = s % length
    lo = int(np.searchsorted(line[:, 0], arc, side="right") - 1)
    lo = min(max(lo, 0), len(line) - 2)
    factor = (arc - line[lo, 0]) / (line[lo + 1, 0] - line[lo, 0])
    return line[lo] * (1 - factor) + line[lo + 1] * factor


@njit(cache=True)
def dynamics(z: np.ndarray, u: np.ndarray, line: np.ndarray, length: float) -> np.ndarray:
    g = geometry(z[0], line, length)
    s, n, e, vx, vy, yaw_rate, steer = z
    mass, inertia, lf, lr, wheelbase = 1550.0, 2400.0, 1.30, 1.66, 2.96
    vx_safe = max(vx, 3.0)
    alpha_f = steer - np.arctan2(vy + lf * yaw_rate, vx_safe)
    alpha_r = -np.arctan2(vy - lr * yaw_rate, vx_safe)
    # Saturated axle forces are a simplified, uncalibrated dynamic AV-21 proxy.
    fy_front = min(max(185000.0 * alpha_f, -mass * 24.0 * lr / wheelbase),
                   mass * 24.0 * lr / wheelbase)
    fy_rear = min(max(195000.0 * alpha_r, -mass * 24.0 * lf / wheelbase),
                  mass * 24.0 * lf / wheelbase)
    ds = (vx * np.cos(e) - vy * np.sin(e)) / max(1 - g[4] * n, 0.3)
    return np.array([ds, vx * np.sin(e) + vy * np.cos(e), yaw_rate - g[4] * ds,
                     u[0] + yaw_rate * vy - fy_front * np.sin(steer) / mass,
                     (fy_front * np.cos(steer) + fy_rear) / mass - yaw_rate * vx,
                     (lf * fy_front * np.cos(steer) - lr * fy_rear) / inertia,
                     u[1]])


@njit(cache=True)
def step(z: np.ndarray, u: np.ndarray, line: np.ndarray, length: float, dt: float = DT) -> np.ndarray:
    u = np.array((min(max(u[0], -8.0), 3.0), min(max(u[1], -0.35), 0.35)))
    out = z.copy()
    for _ in range(2):
        k1 = dynamics(out, u, line, length)
        k2 = dynamics(out + dt / 4 * k1, u, line, length)
        out += dt / 2 * k2
        out[3] = min(max(out[3], 2.0), 90.0)
        out[6] = min(max(out[6], -0.25), 0.25)
    return out


@njit(cache=True)
def gaps(z: np.ndarray, obs: np.ndarray, time: float, line: np.ndarray, length: float):
    g = geometry(z[0], line, length)
    # Conservative projected footprint (world-space SAT is checked on execution).
    envelope = 1.8 + 2.3 * abs(np.sin(z[2]))
    left = g[7] - envelope - z[1]
    right = z[1] - g[6] - envelope
    h = np.empty(len(obs))
    for j in range(len(obs)):
        longitudinal = z[0] - (obs[j, 0] + obs[j, 2] * time)
        if longitudinal > length / 2:
            longitudinal -= length
        elif longitudinal < -length / 2:
            longitudinal += length
        # Ellipse must contain the entire combined 4.6 m × 2 m car footprint,
        # including its corners; 5.6 × 2.8 alone would miss diagonal contacts.
        long_axis = 6.9 + 2.3 * abs(np.sin(z[2]))
        lat_axis = 3.3 + 2.3 * abs(np.sin(z[2]))
        h[j] = (longitudinal / long_axis) ** 2 + ((z[1] - obs[j, 1]) / lat_axis) ** 2 - 1
    return left, right, h


@njit(cache=True)
def forecast_obstacles(obs, line, length, count):
    future = np.empty((count + 1, len(obs), 4))
    future[0] = obs
    for j in range(len(obs)):
        pace = obs[j, 2] / max(geometry(obs[j, 0], line, length)[5], 0.1)
        for k in range(count):
            future[k + 1, j] = future[k, j]
            speed = pace * geometry(future[k, j, 0], line, length)[5]
            future[k + 1, j, 2] = speed
            future[k + 1, j, 0] += DT * speed
    return future


@njit(cache=True)
def rollout(z: np.ndarray, controls: np.ndarray, obs: np.ndarray, target: float,
            shield: bool, line: np.ndarray, length: float):
    states = np.empty((len(controls) + 1, 7))
    states[0] = z
    cost = 0.0
    future = forecast_obstacles(obs, line, length, len(controls))
    _, _, old_h = gaps(z, obs, 0.0, line, length)
    for k in range(len(controls)):
        z = step(z, controls[k], line, length)
        states[k + 1] = z
        g = geometry(z[0], line, length)
        vref = g[8]
        left, right, h = gaps(z, future[k + 1], 0.0, line, length)
        cost += 1.2 * (z[1] - target) ** 2 + 20 * z[2] ** 2
        cost += 0.16 * (z[3] - vref) ** 2 + 0.012 * controls[k, 0] ** 2
        cost += 1.5 * controls[k, 1] ** 2 + 0.12 * z[5] ** 2
        if left < 0 or right < 0:
            cost += 15000 + 10000 * (min(left, right)) ** 2
        for j in range(len(obs)):
            if h[j] < 0:
                cost += 20000 + 10000 * h[j] ** 2
            if shield and old_h[j] < 15:
                residual = h[j] - 0.5 * old_h[j]
                if residual < 0:
                    cost += 50 * residual ** 2
        old_h = h
    return cost, states


@njit(cache=True)
def batch(z, us, obs, target, shield, line, length):
    costs = np.empty(len(us))
    for j in range(len(us)):
        costs[j], _ = rollout(z, us[j], obs, target, shield, line, length)
    return costs


def constraints(z, controls, obs, line, length, count: int, cbf: bool):
    trajectory = np.empty((len(controls) + 1, 7))
    trajectory[0] = z
    margins: list[float] = []
    future = forecast_obstacles(obs, line, length, len(controls))
    _, _, previous_h = gaps(z, obs, 0.0, line, length)
    for k, action in enumerate(controls):
        z = step(z, action, line, length)
        trajectory[k + 1] = z
        left, right, h = gaps(z, future[k + 1], 0.0, line, length)
        margins.extend((left, right))
        margins.extend(h)
        if cbf and k < count:
            margins.extend(h[j] - 0.5 * previous_h[j] if previous_h[j] < 15 else 1e5
                           for j in range(len(obs)))
        previous_h = h
    return np.array(margins), trajectory


def accepted(z, controls, obs, track: Track, cbf: bool, prefix: int = 6):
    values, traj = constraints(z, controls, obs, track.geometry, track.length, prefix, cbf)
    return bool(np.all(values >= 0)), traj


@dataclass
class SideDecision:
    side: int = 0
    target: float = 0.0
    solve_ms: float = 0.0
    feasible: bool = True


def miqp_side(track: Track, state: np.ndarray, obstacles: np.ndarray, previous: SideDecision) -> SideDecision:
    start = perf_counter()
    plan_dt, n_steps = 0.25, 18
    cur = float(state[0])
    active = []
    for obs in obstacles:
        ahead = (obs[0] - cur + track.length / 2) % track.length - track.length / 2
        if -12 < ahead < 180:
            active.append((ahead, obs))
    if not active:
        return SideDecision(solve_ms=(perf_counter() - start) * 1000)

    model = gp.Model("SOfAR_side_MIQP")
    model.Params.OutputFlag = 0
    model.Params.Threads = 1
    model.Params.TimeLimit = 0.025
    model.Params.MIPGap = 0.15
    d = model.addVars(n_steps, lb=-20.0, ub=20.0)
    binaries = model.addVars(len(active), vtype=gp.GRB.BINARY)
    slacks = []
    model.addConstr(d[0] >= state[1] - 0.6)
    model.addConstr(d[0] <= state[1] + 0.6)
    for k in range(n_steps):
        arc = cur + (k + 1) * plan_dt * max(float(state[3]), 15.0)
        width = track.at(arc)
        model.addConstr(d[k] >= width[6] + 1.9)
        model.addConstr(d[k] <= width[7] - 1.9)
        if k:
            model.addConstr(d[k] - d[k - 1] <= 0.85)
            model.addConstr(d[k - 1] - d[k] <= 0.85)
        for j, (ahead, obs) in enumerate(active):
            predicted = ahead + obs[2] * (k + 1) * plan_dt
            if abs((k + 1) * plan_dt * state[3] - predicted) < 5.5:
                slack = model.addVar(lb=0.0)
                slacks.append(slack)
                model.addConstr(d[k] <= obs[1] - 2.8 + 50 * binaries[j] + slack)
                model.addConstr(d[k] >= obs[1] + 2.8 - 50 * (1 - binaries[j]) - slack)
    if not slacks:
        return SideDecision(solve_ms=(perf_counter() - start) * 1000)
    objective = gp.quicksum(2.0 * d[k] * d[k] for k in range(n_steps))
    objective += gp.quicksum(180 * (d[k] - d[k - 1]) * (d[k] - d[k - 1])
                             for k in range(1, n_steps))
    objective += gp.quicksum(50000 * s for s in slacks)
    if previous.side:
        nearest = min(range(len(active)), key=lambda j: active[j][0])
        objective += 300 * (binaries[nearest] - (1 if previous.side == 1 else 0)) ** 2
    model.setObjective(objective, gp.GRB.MINIMIZE)
    model.optimize()
    elapsed = (perf_counter() - start) * 1000
    if model.SolCount == 0:
        return SideDecision(solve_ms=elapsed, feasible=False)
    j = min(range(len(active)), key=lambda j: max(active[j][0], 0))
    side = 1 if binaries[j].X > 0.5 else -1
    target = np.clip(active[j][1][1] + side * 3.6, track.at(cur)[6] + 1.4,
                     track.at(cur)[7] - 1.4)
    if any(slack.X > 0.3 for slack in slacks):
        return SideDecision(solve_ms=elapsed, feasible=False)
    return SideDecision(side, float(target), elapsed)


def guide(z: np.ndarray, obstacles: np.ndarray, decision: SideDecision,
          track: Track, directional: bool) -> np.ndarray:
    return _guide(z, obstacles, decision.side, decision.target, track.geometry, track.length, directional)


@njit(cache=True)
def _guide(z, obstacles, side, lateral_target, line, length, directional):
    actions = np.empty((H, 2))
    predicted = z.copy()
    future = forecast_obstacles(obstacles, line, length, H)
    for k in range(H):
        g = geometry(predicted[0], line, length)
        target = min(max(lateral_target if side else 0.0, g[6] + 1.9), g[7] - 1.9)
        field_s, field_n = 0.0, 0.0
        if directional:
            for o in future[k]:
                ds = (predicted[0] - o[0] + length / 2) % length - length / 2
                dn = predicted[1] - o[1]
                rho = max(np.sqrt((ds / 16.0) ** 2 + (dn / 3.3) ** 2), 0.2)
                grad = np.array([ds / 256, dn / 10.89]) / rho
                if rho < 3.0:
                    rep = min(8.0, 3.0 * (1 / rho - 1 / 3) / rho ** 2)
                    field_s += rep * grad[0]
                    field_n += rep * grad[1]
                if -55 < ds < 0 and side:
                    normal = grad / max(np.linalg.norm(grad), 1e-6)
                    activate = min(max((55 + ds) / 35, 0.0), 1.0)
                    # R_{-sigma*pi/2} n gives positive lateral guidance for left passing.
                    field_s += 2.5 * activate * side * normal[1]
                    field_n -= 2.5 * activate * side * normal[0]
            road_l = max(g[7] - 1.8 - predicted[1], 0.5)
            road_r = max(predicted[1] - g[6] - 1.8, 0.5)
            field_n += 0.3 / road_r - 0.3 / road_l
        lookahead = max(9.0, float(predicted[3]) * 0.45)
        desired_steer = min(max(np.arctan(2.96 * g[4]) + 2 * 2.96 * (target - predicted[1]) /
                                lookahead ** 2 - 2 * 2.96 * predicted[2] / lookahead +
                                2.96 * min(max(field_n, -4.0), 4.0) / max(predicted[3] ** 2, 100),
                                -0.22), 0.22)
        v_target = float(g[8])
        for o in future[k]:
            ds = (o[0] - predicted[0] + length / 2) % length - length / 2
            if 0 < ds < 120 and abs(predicted[1] - o[1]) < 3.3:
                v_target = min(v_target, np.sqrt(o[2] ** 2 + 14 * max(ds - 12, 0)))
        actions[k, 0] = min(max(0.6 * (v_target - predicted[3]) + min(max(field_s, -2.0), 2.0), -8.0), 3.0)
        actions[k, 1] = min(max((desired_steer - predicted[6]) / 0.28, -0.35), 0.35)
        predicted = step(predicted, actions[k], line, length)
    return actions


def repair(z, proposal, obs, track: Track):
    # Sequential convexification over two 2-step blocks: only one small QP per iteration.
    candidate = proposal.copy()
    for _ in range(2):
        base, _ = constraints(z, candidate, obs, track.geometry, track.length, 6, True)
        if np.min(base) >= 0:
            return candidate
        jac = np.empty((len(base), 4))
        for j in range(4):
            varied = candidate.copy()
            varied[j // 2 * 2:j // 2 * 2 + 2, j % 2] += 0.015
            values, _ = constraints(z, varied, obs, track.geometry, track.length, 6, True)
            jac[:, j] = (values - base) / 0.015
        lo = np.array([-8 - candidate[0, 0], -0.35 - candidate[0, 1],
                       -8 - candidate[2, 0], -0.35 - candidate[2, 1]])
        hi = np.array([3 - candidate[0, 0], 0.35 - candidate[0, 1],
                       3 - candidate[2, 0], 0.35 - candidate[2, 1]])
        lo = np.maximum(lo, [-2.0, -0.2, -2.0, -0.2])
        hi = np.minimum(hi, [2.0, 0.2, 2.0, 0.2])
        A = sparse.vstack([sparse.csc_matrix(jac), sparse.eye(4, format="csc")], format="csc")
        qp = osqp.OSQP()
        qp.setup(P=sparse.diags([0.5, 15, 0.5, 15], format="csc"), q=np.zeros(4), A=A,
                 l=np.r_[-base, lo], u=np.r_[np.full(len(base), np.inf), hi],
                 verbose=False, max_iter=200, polishing=False)
        result = qp.solve(raise_error=False)
        if result.info.status not in {"solved", "solved inaccurate"}:
            return None
        for block in range(2):
            candidate[2 * block:2 * block + 2] += result.x[block * 2:block * 2 + 2]
        candidate = np.clip(candidate, [-8.0, -0.35], [3.0, 0.35])
        if accepted(z, candidate, obs, track, True)[0]:
            return candidate
    return None


class MPPI:
    def __init__(self, method: str, track: Track, seed: int):
        if method not in {"B0", "B1", "B2"}:
            raise ValueError(method)
        self.method, self.track = method, track
        self.rng = np.random.default_rng(seed)
        self.nominal = np.zeros((H, 2))
        self.decision = SideDecision()
        self.cycle = 0
        self.repairs = 0
        self.repair_success = 0
        self.fallbacks = 0
        self.repair_ms = 0.0
        self.target = 0.0

    def control(self, state: np.ndarray, obstacles: np.ndarray):
        start = perf_counter()
        miqp_ms = 0.0
        if self.cycle % 5 == 0:
            proposed = miqp_side(self.track, state, obstacles, self.decision)
            miqp_ms = proposed.solve_ms
            if proposed.feasible:
                self.decision = proposed
            elif self.decision.side:
                self.decision.feasible = False
                self.decision.solve_ms = miqp_ms
            else:
                self.decision = proposed
        directional = self.method != "B0"
        shield = self.method != "B2"
        self.target += float(np.clip(self.decision.target - self.target, -0.3, 0.3))
        smooth = SideDecision(self.decision.side, self.target)
        gaussian = guide(state, obstacles, smooth, self.track, False)
        assist = guide(state, obstacles, smooth, self.track, True) if directional else gaussian
        shifted = np.vstack((self.nominal[1:], self.nominal[-1]))
        reference = 0.3 * shifted + 0.7 * gaussian
        noise = self.rng.standard_normal((K, H, 2)) * np.array([1.9, 0.14])
        candidates = np.clip(reference + noise, [-8, -0.35], [3, 0.35])
        if directional:
            candidates[64:127] = np.clip(assist + noise[64:127], [-8, -0.35], [3, 0.35])
            candidates[127] = assist  # atomic proposal; equal per-sample mixture mass
        costs = batch(state, candidates, obstacles, self.target,
                      shield, self.track.geometry, self.track.length)
        softmin = float(np.min(costs))
        weights = np.exp(-np.clip((costs - softmin) / 45.0, 0, 700))
        weights /= weights.sum()
        sequence = np.einsum("k,kij->ij", weights, candidates)
        ok, _ = accepted(state, sequence, obstacles, self.track, shield)
        if not ok and shield:
            self.repairs += 1
            repair_start = perf_counter()
            amended = repair(state, sequence, obstacles, self.track)
            self.repair_ms += (perf_counter() - repair_start) * 1000
            if amended is not None and accepted(state, amended, obstacles, self.track, shield)[0]:
                sequence = amended
                ok = True
                self.repair_success += 1
        if not ok:
            self.fallbacks += 1
            # Never execute an unsafe weighted average just because sampled inputs were safe.
            for j in np.argsort(costs):
                if accepted(state, candidates[j], obstacles, self.track, shield)[0]:
                    sequence = candidates[j]
                    ok = True
                    break
        if not ok:
            sequence = gaussian.copy()
            sequence[:, 0] = -8.0
            # No safe action found: explicitly report unrecoverable state.
            ok, _ = accepted(state, sequence, obstacles, self.track, False)
        self.nominal = sequence.copy()
        self.cycle += 1
        elapsed_ms = (perf_counter() - start) * 1000
        return sequence[0].copy(), elapsed_ms, miqp_ms, ok
