import numpy as np
import pytest

from experiments.controller import H, K, MPPI, SideDecision, accepted, constraints, forecast_obstacles, gaps, guide, miqp_side, step
from experiments.run import dataset_hashes, duration_steps, initial_state, run_episode, summarize
from experiments.track import Track, collision, off_track


@pytest.mark.parametrize("name", ["LS", "KS", "IMS", "Monza"])
def test_published_geometry_and_start(name):
    track = Track.load(name)
    assert track.length > 2000
    for seed in (101, 102, 201, 205):
        state = initial_state(track, seed)
        assert not off_track(track, state)
        assert all(not collision(track, state, obs)
                   for obs in track.initial_obstacles(0.3, float(state[0])))
        assert abs(track.project(track.to_world(state[0], state[1]))[1] - state[1]) < 0.3


def test_exact_vehicle_geometry_rejects_overlap_and_track_exit():
    track = Track.load("LS")
    state = initial_state(track, 201)
    obstacle = np.array([state[0], state[1], 10.0, 0.0])
    assert collision(track, state, obstacle)
    obstacle[0] += 30
    assert not collision(track, state, obstacle)
    state[1] = track.at(float(state[0]))[7] - 0.2
    assert off_track(track, state)


def test_dynamic_steering_rate_has_actuator_delay():
    track = Track.load("LS")
    state = initial_state(track, 201)
    a = step(state, np.array([0.0, 0.35]), track.geometry, track.length)
    assert a[6] > state[6]
    assert a[6] < 0.25
    assert np.isfinite(a).all()


def test_miqp_side_and_signed_tangential_candidate():
    track = Track.load("LS")
    state = initial_state(track, 201)
    obs = np.array([[state[0] + 40, 0.0, 10.0, 0.0]])
    decision = miqp_side(track, state, obs, SideDecision())
    assert decision.feasible
    assert decision.side in (-1, 1)
    assert decision.side * decision.target > 0
    assisted = guide(state, obs, decision, track, True)
    neutral = guide(state, obs, SideDecision(), track, False)
    assert assisted.shape == neutral.shape == (H, 2)
    assert decision.side * (assisted[0, 1] - neutral[0, 1]) > 0


def test_unsafely_colliding_weighted_proposal_is_not_accepted():
    track = Track.load("LS")
    state = initial_state(track, 201)
    obs = np.array([[state[0] + 10, 0.0, 1.0, 0.0]])
    controls = np.zeros((H, 2))
    assert not accepted(state, controls, obs, track, True)[0]
    assert not accepted(state, controls, obs, track, False)[0]


def test_all_methods_use_identical_rollout_budget_and_seeds():
    track = Track.load("LS")
    state = initial_state(track, 201)
    obs = track.initial_obstacles(0.3, float(state[0]))
    assert K == 128 and H == 20
    for method in ("B0", "B1", "B2"):
        controller = MPPI(method, track, seed=201)
        action, ms, _, _ = controller.control(state, obs)
        assert action.shape == (2,)
        assert np.isfinite(action).all()
        assert ms > 0
        assert controller.cycle == 1


def test_episode_records_failures_and_paired_state():
    track = Track.load("LS")
    a = run_episode(track, 0.3, 201, "B0", max_time=0.3)
    b = run_episode(track, 0.3, 201, "B1", max_time=0.3)
    assert a["cycles"] == b["cycles"] == 3
    assert a["numba_warmup_excluded"]
    assert a["sim_s"] == b["sim_s"] == 0.3
    summary = summarize([a, b])
    assert len(summary) == 2
    assert all(row["n"] == 1 for row in summary)


def test_diagonal_contact_is_inside_safety_ellipse():
    track = Track.load("LS")
    z = initial_state(track, 201)
    obs = np.array([[z[0] + 4.5, z[1] + 1.9, z[3], 0.0]])
    assert collision(track, z, obs[0])
    assert gaps(z, obs, 0.0, track.geometry, track.length)[2][0] < 0


def test_forecast_matches_executed_opponent_velocity_profile():
    track = Track.load("LS")
    obs = track.initial_obstacles(0.65, 600.0)
    predicted = forecast_obstacles(obs, track.geometry, track.length, H)
    for k in range(H):
        track.advance_obstacles(obs, 0.65, 0.1)
        assert np.allclose(obs, predicted[k + 1])


@pytest.mark.parametrize("name", ["LS", "Monza"])
def test_forecast_remains_consistent_after_advance_wrap_and_recycle(name):
    track = Track.load(name)
    pace = 0.65
    obs = track.initial_obstacles(pace, track.length - 50)
    for _ in range(23):
        track.advance_obstacles(obs, pace, 0.1)
    track.recycle(track.length + 100, obs, 0)
    for j in range(4):
        assert np.isclose(obs[j, 2], pace * track.at(obs[j, 0])[5])
    predicted = forecast_obstacles(obs, track.geometry, track.length, H)
    for k in range(H):
        track.advance_obstacles(obs, pace, 0.1)
        assert np.allclose(obs, predicted[k + 1])


def test_miqp_targets_imminent_crossing_not_unconstrained_near_car():
    track = Track.load("LS")
    state = initial_state(track, 201)
    obs = np.array([[state[0] + 10, -1.0, 100.0, 0.0],
                    [state[0] + 70, 2.0, 7.0, 0.0]])
    previous = SideDecision(1, 2.6, obstacle_idx=0)
    decision = miqp_side(track, state, obs, previous)
    assert decision.feasible and decision.side in (-1, 1)
    assert decision.obstacle_idx == 1
    assert decision.target == pytest.approx(np.clip(obs[1, 1] + decision.side * 3.6,
                                                   track.at(state[0])[6] + 1.4,
                                                   track.at(state[0])[7] - 1.4))


def test_shield_finite_difference_dimensions_do_not_change():
    track = Track.load("LS")
    z = initial_state(track, 201)
    obs = track.initial_obstacles(0.3, float(z[0]))
    controls = np.zeros((H, 2))
    base, _ = constraints(z, controls, obs, track.geometry, track.length, 6, True)
    controls[:2, 1] += 0.015
    varied, _ = constraints(z, controls, obs, track.geometry, track.length, 6, True)
    assert base.shape == varied.shape == (144,)


def test_resumed_csv_summary_preserves_false_success():
    row = run_episode(Track.load("LS"), 0.3, 201, "B0", max_time=0.1)
    row["success"] = "False"
    row["failure"] = "collision"
    summary = summarize([row])[0]
    assert summary["success_rate"] == 0.0 and summary["collisions"] == 1


def test_cli_resumes_without_duplicate_episodes(tmp_path):
    import subprocess
    import sys
    import csv

    command = [sys.executable, "-m", "experiments.run", "--suite", "custom",
               "--tracks", "LS", "--pace", "0.3", "--seeds", "201", "--methods", "B0",
               "--max-time", "0.1", "--output", str(tmp_path)]
    subprocess.run(command, check=True, capture_output=True)
    subprocess.run(command, check=True, capture_output=True)
    with (tmp_path / "episodes.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 1
    command[command.index("0.1")] = "0.2"
    assert subprocess.run(command, capture_output=True).returncode != 0


@pytest.mark.parametrize("duration", [0.01, 0.05, 0.15, 0.25, float("nan"), float("inf")])
def test_reject_non_grid_or_nonfinite_durations(duration):
    with pytest.raises(ValueError, match="max-time"):
        duration_steps(duration)


def test_duration_and_episode_accept_exact_grid():
    assert duration_steps(0.3) == 3
    with pytest.raises(ValueError, match="max-time"):
        run_episode(Track.load("LS"), 0.3, 201, "B0", 0.01)


def test_dataset_hashes_include_ls_dependency_and_detect_edits(tmp_path):
    for folder, prefix in (("racelines", "raceline"), ("boundaries", "boundaries")):
        (tmp_path / folder).mkdir()
        for track in ("KS", "LS"):
            (tmp_path / folder / f"{prefix}_{track}.csv").write_text("original")
    before = dataset_hashes(["KS"], tmp_path)
    assert len(before) == 4 and "racelines/raceline_LS.csv" in before
    (tmp_path / "racelines/raceline_LS.csv").write_text("changed")
    assert before != dataset_hashes(["KS"], tmp_path)


def test_cli_regenerates_missing_or_corrupt_trajectories_without_duplicate_rows(tmp_path):
    import subprocess
    import sys
    import csv
    import json

    command = [sys.executable, "-m", "experiments.run", "--suite", "custom", "--tracks", "LS",
               "--pace", "0.3", "--seeds", "201", "--methods", "B0", "--max-time", "0.1",
               "--output", str(tmp_path)]
    subprocess.run(command, check=True, capture_output=True)
    command.append("--trajectories")
    p = tmp_path / "trajectory_LS_0.3_201_B0.npz"
    for _ in range(2):
        subprocess.run(command, check=True, capture_output=True)
        assert np.load(p)["samples"].shape == (1, 19)
        with (tmp_path / "episodes.csv").open() as f:
            assert len(list(csv.DictReader(f))) == 1
        p.write_text("corrupt file")
    manifest = tmp_path / "manifest.json"
    data = json.loads(manifest.read_text())
    data["dataset_sha256"]["racelines/raceline_LS.csv"] = "changed"
    manifest.write_text(json.dumps(data))
    result = subprocess.run(command, capture_output=True)
    assert result.returncode != 0 and b"dataset_sha256" in result.stderr


def test_report_audits_and_aggregates_real_cycles(tmp_path):
    import subprocess
    import sys
    import json
    from experiments.report import report

    raw, out = tmp_path / "raw", tmp_path / "report"
    subprocess.run([sys.executable, "-m", "experiments.run", "--suite", "custom", "--tracks", "LS",
                    "--pace", "0.3", "--seeds", "201", "--max-time", "0.1", "--output", str(raw),
                    "--trajectories"], check=True, capture_output=True)
    report(raw, out)
    totals = json.loads((out / "totals.json").read_text())
    assert all(r["n"] == r["completions"] == r["cycles"] == 1 for r in totals)
    assert all(r["physical_end_contacts"] == 0 for r in totals)
    assert "0.1 s 完成" in (out / "REPORT.md").read_text()
    path = raw / "trajectory_LS_0.3_201_B1.npz"
    with np.load(path) as data:
        values = {k: data[k] for k in data.files}
    values["samples"] = np.empty((0, 19))
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="Trajectory rows"):
        report(raw, out)
