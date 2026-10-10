"""Run paired headless closed-loop experiments; output resumable raw episode rows.

Example: python -m experiments.run --suite development --output batch_results/mppi_ls
"""

import argparse
import csv
import hashlib
import json
import platform
import os
from importlib.metadata import version
from datetime import datetime, timezone
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from experiments.controller import DT, H, K, MPPI, step
from experiments.track import Track, collision, off_track

FIELDS = ["track", "pace", "seed", "method", "sim_s", "success", "failure",
          "laps", "overtakes", "cycles", "mean_ms", "p95_ms", "p99_ms",
          "miqp_mean_ms", "miqp_p95_ms", "repair_attempts", "repair_successes",
          "fallbacks", "unrecoverable", "repair_total_ms", "repair_amortized_ms",
          "deadline_overruns", "numba_warmup_excluded"]


def duration_steps(max_time: float) -> int:
    if not np.isfinite(max_time) or max_time < DT:
        raise ValueError(f"max-time must be at least {DT}s and an integer multiple of it")
    steps = round(max_time / DT)
    if not np.isclose(steps * DT, max_time, rtol=0, atol=1e-9):
        raise ValueError(f"max-time must be an integer multiple of {DT}s")
    return steps


def dataset_hashes(tracks: list[str], root: Path | None = None) -> dict[str, str]:
    root = root or Path(__file__).resolve().parents[1]
    paths = [f"{folder}/{prefix}_{name}.csv" for name in sorted(set(tracks) | {"LS"})
             for folder, prefix in (("racelines", "raceline"), ("boundaries", "boundaries"))]
    return {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}


def trajectory_complete(path: Path, cycles: int) -> bool:
    try:
        with np.load(path) as data:
            return data["samples"].shape == (cycles, 19) and len(data["cycle_ms"]) == cycles
    except (OSError, ValueError, KeyError, EOFError, BadZipFile):
        return False


def initial_state(track: Track, seed: int) -> np.ndarray:
    # Deterministic randomized, paired ego starts near the author's LS index 20.
    rng = np.random.default_rng(seed)
    start_arc = track.geometry[20, 0] + rng.uniform(-4, 4)
    return np.array([start_arc, rng.uniform(-0.15, 0.15),
                     rng.uniform(-0.006, 0.006), 35 + rng.uniform(-1.5, 1.5),
                     0.0, 0.0, 0.0], dtype=float)


def run_episode(track: Track, pace: float, seed: int, method: str, max_time: float,
                trajectory_path: Path | None = None) -> dict:
    steps = duration_steps(max_time)
    ego = initial_state(track, seed)
    obstacles = track.initial_obstacles(pace, float(ego[0]))
    if off_track(track, ego) or any(collision(track, ego, obs) for obs in obstacles):
        raise ValueError(f"Invalid initial state {track.name} {pace} {seed}")
    controller = MPPI(method, track, seed)
    cycle_ms, miqp_ms = [], []
    traj, overtakes = [], 0
    failure = ""
    start_arc = float(ego[0])
    unrecoverable = 0
    for i in range(steps):
        action, ms, high_ms, safe = controller.control(ego, obstacles)
        cycle_ms.append(ms)
        if high_ms:
            miqp_ms.append(high_ms)
        if not safe:
            unrecoverable += 1
        ego = step(ego, action, track.geometry, track.length)
        track.advance_obstacles(obstacles, pace, DT)
        if trajectory_path is not None:
            traj.append([(i + 1) * DT, *ego, *action, controller.decision.side,
                         *obstacles[:, 0], *obstacles[:, 1]])
        if not np.all(np.isfinite(ego)):
            failure = "nonfinite_state"
        elif off_track(track, ego):
            failure = "track_boundary_violation"
        elif any(collision(track, ego, obs) for obs in obstacles):
            failure = "collision"
        if failure:
            break
        for j, obs in enumerate(obstacles):
            dist = (obs[0] - ego[0] + track.length / 2) % track.length - track.length / 2
            if 20 <= dist <= 180 and obs[3] == 0:
                obs[3] = 1
            elif dist < -6 and obs[3] == 1:
                overtakes += 1
                track.recycle(ego[0], obstacles, j)
    if trajectory_path is not None:
        np.savez_compressed(trajectory_path, samples=np.asarray(traj), cycle_ms=cycle_ms,
                            miqp_ms=miqp_ms,
                            columns=np.array(["sim_s", "s", "n", "heading_error", "vx", "vy",
                                              "yaw_rate", "steer", "a", "steer_rate", "miqp_side",
                                              "obs_s_0", "obs_s_1", "obs_s_2", "obs_s_3",
                                              "obs_n_0", "obs_n_1", "obs_n_2", "obs_n_3"]))
    return dict(track=track.name, pace=pace, seed=seed, method=method,
                sim_s=round(len(cycle_ms) * DT, 2), success=not bool(failure), failure=failure,
                laps=(ego[0] - start_arc) / track.length, overtakes=overtakes,
                cycles=len(cycle_ms), mean_ms=float(np.mean(cycle_ms)),
                p95_ms=float(np.percentile(cycle_ms, 95)),
                p99_ms=float(np.percentile(cycle_ms, 99)),
                miqp_mean_ms=float(np.mean(miqp_ms)) if miqp_ms else 0.0,
                miqp_p95_ms=float(np.percentile(miqp_ms, 95)) if miqp_ms else 0.0,
                repair_attempts=controller.repairs, repair_successes=controller.repair_success,
                fallbacks=controller.fallbacks, unrecoverable=unrecoverable,
                repair_total_ms=controller.repair_ms,
                repair_amortized_ms=controller.repair_ms / len(cycle_ms),
                deadline_overruns=sum(ms > 100 for ms in cycle_ms),
                numba_warmup_excluded=True)


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["track"], row["method"]), []).append(row)
    return [dict(track=track, method=method, n=len(items),
                 success_rate=sum(str(r["success"]) in {"True", "true"} for r in items) / len(items),
                 simulation_seconds=sum(float(r["sim_s"]) for r in items),
                 laps=sum(float(r["laps"]) for r in items),
                 overtakes=sum(int(r["overtakes"]) for r in items),
                 mean_cycle_ms=float(np.mean([float(r["mean_ms"]) for r in items])),
                 max_episode_p95_ms=float(np.max([float(r["p95_ms"]) for r in items])),
                 collisions=sum(r["failure"] == "collision" for r in items),
                 boundary_violations=sum(r["failure"] == "track_boundary_violation" for r in items))
            for (track, method), items in sorted(grouped.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["development", "final", "custom"], default="development")
    parser.add_argument("--tracks", nargs="+", choices=["LS", "KS", "IMS", "Monza"])
    parser.add_argument("--pace", nargs="+", type=float)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--methods", nargs="+", choices=["B0", "B1", "B2"])
    parser.add_argument("--max-time", type=float, default=200.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trajectories", action="store_true")
    args = parser.parse_args()
    presets = {"development": (["LS"], [0.3, 0.65], list(range(101, 106))),
               "final": (["KS", "IMS", "LS", "Monza"], [0.1, 0.3, 0.5, 0.65, 0.8],
                         list(range(201, 206))), "custom": ([], [], [])}
    tracks, pace_levels, seeds = presets[args.suite]
    tracks, pace_levels, seeds = args.tracks or tracks, args.pace or pace_levels, args.seeds or seeds
    if not tracks or not pace_levels or not seeds or args.max_time <= 0:
        parser.error("A valid track, pace, seeds, and positive max-time are required")
    try:
        duration_steps(args.max_time)
    except ValueError as exc:
        parser.error(str(exc))
    if any(not np.isfinite(p) or p < 0 or p > 1 for p in pace_levels):
        parser.error("Pace is a factor of the published raceline velocity, between 0 and 1")
    methods = args.methods or ["B0", "B1", "B2"]
    args.output.mkdir(parents=True, exist_ok=True)
    results_file = args.output / "episodes.csv"
    current = list(csv.DictReader(results_file.open())) if results_file.exists() else []
    keys = {(r["track"], float(r["pace"]), int(r["seed"]), r["method"]) for r in current}
    if len(keys) != len(current):
        raise ValueError("Duplicate existing episode keys")
    # Warm the JIT outside timed episodes; no first-cycle compilation in latency metrics.
    warm = Track.load(tracks[0])
    z = initial_state(warm, seeds[0])
    warm_controller = MPPI("B0", warm, seeds[0])
    warm_controller.control(z, warm.initial_obstacles(pace_levels[0], z[0]))
    sources = [Path(__file__).with_name(name) for name in ("controller.py", "track.py", "run.py")]
    digest = hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()
    manifest = dict(suite=args.suite, tracks=tracks, pace=pace_levels, seeds=seeds,
                    methods=methods, max_time=args.max_time, plant_dt=DT, horizon=H,
                    candidates=K, created_utc=datetime.now(timezone.utc).isoformat(),
                    model="uncalibrated_dynamic_bicycle; not author's acados MPC",
                    obstacle_initialization="author LS indices as relative arc offsets; 4 cars",
                    completion=f"{args.max_time:g}s without collision or track-boundary crossing",
                    source_sha256=digest, dataset_sha256=dataset_hashes(tracks),
                    python=platform.python_version(), cpu_count=os.cpu_count(),
                    dependencies={p: version(p) for p in ("numpy", "scipy", "numba", "gurobipy", "osqp", "matplotlib")},
                    ego_speed_reference="min(0.8 raceline,39m/s,sqrt(16/abs(curvature))); 7m/s² backward braking envelope",
                    repair_timing="QP repair only; does not isolate batch barrier-cost overhead",
                    safety="engineering filter, not a proven CBF invariant; unrecoverable actions explicitly counted",
                    latency="CPU perf_counter includes planning/selection/shield; excludes JIT compilation")
    manifest_path = args.output / "manifest.json"
    if not manifest_path.exists():
        if current:
            raise ValueError("Cannot resume episode rows without their original manifest")
        manifest_path.write_text(json.dumps(manifest, indent=2))
    else:
        old = json.loads(manifest_path.read_text())
        for key in ("suite", "tracks", "pace", "seeds", "methods", "max_time", "source_sha256", "dataset_sha256", "dependencies"):
            if old.get(key) != manifest[key]:
                raise ValueError(f"Resume manifest disagrees on {key}")
    total = len(tracks) * len(pace_levels) * len(seeds) * len(methods)
    for name in tracks:
        track = Track.load(name)
        for pace in pace_levels:
            for seed in seeds:
                for method in methods:
                    key = (name, pace, seed, method)
                    trajectory = args.output / f"trajectory_{name}_{pace:g}_{seed}_{method}.npz" if args.trajectories else None
                    if key in keys:
                        old_row = next(r for r in current if (r["track"], float(r["pace"]), int(r["seed"]), r["method"]) == key)
                        if trajectory is None or trajectory_complete(trajectory, int(old_row["cycles"])):
                            continue
                    row = run_episode(track, pace, seed, method, args.max_time, trajectory)
                    current = [r for r in current if (r["track"], float(r["pace"]), int(r["seed"]), r["method"]) != key]
                    current.append(row)
                    temporary = results_file.with_suffix(".csv.tmp")
                    with temporary.open("w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
                        writer.writeheader()
                        writer.writerows(current)
                    temporary.replace(results_file)
                    keys.add(key)
                    (args.output / "summary.json").write_text(json.dumps(summarize(current), indent=2))
                    print(f"{len(current)}/{total}: {key} success={row['success']} "
                          f"t={row['sim_s']}s laps={row['laps']:.2f} "
                          f"overtakes={row['overtakes']} P95={row['p95_ms']:.1f}ms",
                          flush=True)


if __name__ == "__main__":
    main()
