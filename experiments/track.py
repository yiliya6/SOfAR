"""Published SOfAR geometry and obstacle motion in global arc coordinates."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from matplotlib.path import Path as Polygon
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Track:
    name: str
    length: float
    geometry: np.ndarray  # [s, x, y, yaw_unwrapped, curvature, speed, right, left, preview_speed]
    inner: np.ndarray
    outer: np.ndarray
    tree: cKDTree
    inner_polygon: Polygon
    outer_polygon: Polygon

    @classmethod
    @lru_cache(maxsize=4)
    def load(cls, name: str) -> "Track":
        if name not in {"KS", "IMS", "LS", "Monza"}:
            raise ValueError(f"Unknown SOfAR track: {name}")
        race = np.loadtxt(ROOT / "racelines" / f"raceline_{name}.csv", delimiter=",")
        bounds = np.loadtxt(ROOT / "boundaries" / f"boundaries_{name}.csv", delimiter=",")
        if np.linalg.norm(race[0, :2] - race[-1, :2]) < 1e-6:
            race = race[:-1]
        xy = race[:, :2]
        ds = np.linalg.norm(np.roll(xy, -1, axis=0) - xy, axis=1)
        length = float(ds.sum())
        s = np.r_[0.0, np.cumsum(ds[:-1])]
        tangent = np.roll(xy, -1, axis=0) - np.roll(xy, 1, axis=0)
        yaw = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
        curvature = np.gradient(yaw, s, edge_order=1)
        normals = np.column_stack((-np.sin(yaw), np.cos(yaw)))
        inner, outer = bounds[:, :2], bounds[:, 2:4]
        left = np.einsum("ij,ij->i", inner[cKDTree(inner).query(xy)[1]] - xy, normals)
        right = np.einsum("ij,ij->i", outer[cKDTree(outer).query(xy)[1]] - xy, normals)
        if np.min(left) < 0 or np.max(right) > 0:
            raise ValueError(f"Track {name} boundary directions are inconsistent")
        speed_limit = np.minimum.reduce((0.8 * race[:, 2], np.full(len(s), 39.0),
                                         np.sqrt(16.0 / np.maximum(np.abs(curvature), 0.001))))
        # Circular backward braking envelope; do not discover a tight bend at its entry.
        preview = speed_limit.copy()
        for _ in range(2):
            for i in range(len(preview) - 1, -1, -1):
                nxt = (i + 1) % len(preview)
                preview[i] = min(preview[i], np.sqrt(preview[nxt] ** 2 + 14.0 * ds[i]))
        geom = np.column_stack((s, xy, yaw, curvature, race[:, 2], right, left, preview))
        geom = np.vstack((geom, np.r_[length, xy[0], yaw[0] + 2 * np.pi * round((yaw[-1] - yaw[0]) / (2 * np.pi)), curvature[0], race[0, 2], right[0], left[0], preview[0]]))
        def area(p):
            return abs(np.dot(p[:, 0], np.roll(p[:, 1], 1)) - np.dot(p[:, 1], np.roll(p[:, 0], 1)))
        inside, outside = (inner, outer) if area(inner) < area(outer) else (outer, inner)
        return cls(name, length, geom, inner, outer, cKDTree(xy),
                   Polygon(np.vstack((inside, inside[0]))), Polygon(np.vstack((outside, outside[0]))))

    def at(self, arc: float) -> np.ndarray:
        i = np.searchsorted(self.geometry[:, 0], arc % self.length, side="right") - 1
        first, second = self.geometry[i], self.geometry[i + 1]
        alpha = (arc % self.length - first[0]) / (second[0] - first[0])
        return first * (1 - alpha) + second * alpha

    def to_world(self, arc: float, lateral: float) -> np.ndarray:
        g = self.at(arc)
        return np.array([g[1] - lateral * np.sin(g[3]), g[2] + lateral * np.cos(g[3])])

    def project(self, xy: np.ndarray) -> tuple[float, float]:
        idx = int(self.tree.query(xy)[1])
        near = self.geometry[idx]
        longitudinal = (xy[0] - near[1]) * np.cos(near[3]) + (xy[1] - near[2]) * np.sin(near[3])
        lateral = -(xy[0] - near[1]) * np.sin(near[3]) + (xy[1] - near[2]) * np.cos(near[3])
        return float((near[0] + longitudinal) % self.length), float(lateral)

    def initial_obstacles(self, pace: float, ego_arc: float, ego_start_index: int = 20) -> np.ndarray:
        # Author defaults for LS, translated by arc distance to retain relative spacing on other tracks.
        reference = Track.load("LS") if self.name != "LS" else self
        origins = reference.geometry[[150, 159, 122, 302], 0] - reference.geometry[ego_start_index, 0]
        origin_arc = ego_arc + origins
        return np.array([[arc, offset, pace * self.at(arc)[5], 0.0]
                         for arc, offset in zip(origin_arc, (0.0, 1.0, -1.0, 0.0))], dtype=float)

    def advance_obstacles(self, obstacles: np.ndarray, pace: float, dt: float) -> None:
        for obs in obstacles:
            obs[2] = pace * self.at(obs[0])[5]
            obs[0] += obs[2] * dt

    def front_distance(self, ego_arc: float, obs_arc: float) -> float:
        return (obs_arc - ego_arc) % self.length

    def recycle(self, ego_arc: float, obstacles: np.ndarray, idx: int) -> None:
        # The author's code teleports completed opponents ~200 raceline points ahead.
        # Use its LS arc distance across all four tracks; skip other nearby opponents.
        ls = Track.load("LS")
        distance = ls.geometry[220, 0] - ls.geometry[20, 0]
        candidate = ego_arc + distance
        while any(j != idx and abs((o[0] - candidate + self.length / 2) % self.length - self.length / 2) < 18
                  for j, o in enumerate(obstacles)):
            candidate += 25
        obstacles[idx, 0] = candidate
        obstacles[idx, 3] = 0.0


def vehicle_corners(track: Track, state: np.ndarray, length: float = 4.6, width: float = 2.0) -> np.ndarray:
    center = track.to_world(float(state[0]), float(state[1]))
    yaw = track.at(float(state[0]))[3] + state[2]
    tangent = np.array([np.cos(yaw), np.sin(yaw)])
    normal = np.array([-tangent[1], tangent[0]])
    return np.array([center + a * length / 2 * tangent + b * width / 2 * normal
                     for a in (-1, 1) for b in (-1, 1)])


def collision(track: Track, state: np.ndarray, obstacle: np.ndarray) -> bool:
    a = vehicle_corners(track, state)
    other = np.array([obstacle[0], obstacle[1], 0.0])
    b = vehicle_corners(track, other)
    # Separating axis theorem for two oriented car rectangles.
    for corners in (a, b):
        for j, k in ((0, 2), (0, 1)):
            edge = corners[k] - corners[j]
            axis = np.array([-edge[1], edge[0]])
            if np.max(a @ axis) < np.min(b @ axis) or np.max(b @ axis) < np.min(a @ axis):
                return False
    return True


def off_track(track: Track, state: np.ndarray, margin: float = 0.0) -> bool:
    corners = vehicle_corners(track, state)
    if not np.all(track.outer_polygon.contains_points(corners, radius=-margin)):
        return True
    if np.any(track.inner_polygon.contains_points(corners, radius=margin)):
        return True
    footprint = Polygon(corners[[0, 2, 3, 1, 0]])
    return (footprint.intersects_path(track.inner_polygon, filled=False) or
            footprint.intersects_path(track.outer_polygon, filled=False))
