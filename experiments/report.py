"""Audit raw experiments, aggregate actual cycles, and render reproducible figures."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.track import Track, collision
from experiments.run import DT, dataset_hashes

COLORS = {"B0": "#0072B2", "B1": "#D55E00", "B2": "#009E73"}


def report(source: Path, output: Path):
    manifest = json.loads((source / "manifest.json").read_text())
    root = Path(__file__).parent
    digest = hashlib.sha256(b"".join((root / p).read_bytes() for p in ("controller.py", "track.py", "run.py"))).hexdigest()
    if manifest["source_sha256"] != digest or manifest.get("dataset_sha256") != dataset_hashes(manifest["tracks"]):
        raise ValueError("Audit requires the same frozen source and track data as the experiment")
    with (source / "episodes.csv").open() as f:
        rows = list(csv.DictReader(f))
    expected = {(t, float(p), int(s), m) for t in manifest["tracks"]
                for p in manifest["pace"] for s in manifest["seeds"] for m in manifest["methods"]}
    keys = [(r["track"], float(r["pace"]), int(r["seed"]), r["method"]) for r in rows]
    if set(keys) != expected or len(keys) != len(set(keys)):
        raise ValueError("Incomplete or duplicate experiment matrix; cannot issue a final report")
    output.mkdir(parents=True, exist_ok=True)
    timing: dict[str, list[np.ndarray]] = {m: [] for m in manifest["methods"]}
    miqp_timing: dict[str, list[np.ndarray]] = {m: [] for m in manifest["methods"]}
    end_contacts: dict[str, int] = {m: 0 for m in manifest["methods"]}
    for row in rows:
        if row["success"] == "True" and not np.isclose(float(row["sim_s"]), manifest["max_time"], rtol=0, atol=1e-9):
            raise ValueError("A successful episode did not complete its duration")
        path = source / f"trajectory_{row['track']}_{float(row['pace']):g}_{row['seed']}_{row['method']}.npz"
        data = np.load(path)
        if len(data["cycle_ms"]) != int(row["cycles"]):
            raise ValueError("Latency records do not match actual executed cycles")
        if data["samples"].shape != (int(row["cycles"]), 19) or not np.allclose(data["samples"][:, 0], np.arange(1, int(row["cycles"]) + 1) * DT):
            raise ValueError("Trajectory rows do not match actual executed cycles")
        timing[row["method"]].append(data["cycle_ms"])
        miqp_timing[row["method"]].append(data["miqp_ms"])
        end = data["samples"][-1]
        track = Track.load(row["track"])
        end_contacts[row["method"]] += int(any(collision(track, end[1:8], np.array([end[11+j], end[15+j], 0.0, 0.0]))
                                               for j in range(4)))
    totals: list[dict] = []
    for method in manifest["methods"]:
        selected = [r for r in rows if r["method"] == method]
        latencies = np.concatenate(timing[method])
        high_latencies = np.concatenate(miqp_timing[method])
        repairs = sum(int(r["repair_attempts"]) for r in selected)
        repair_ms = sum(float(r["repair_total_ms"]) for r in selected)
        totals.append(dict(method=method, n=len(selected),
                           completions=sum(r["success"] == "True" for r in selected),
                           collisions=sum(r["failure"] == "collision" for r in selected),
                           boundary_failures=sum(r["failure"] == "track_boundary_violation" for r in selected),
                           physical_end_contacts=end_contacts[method],
                           mean_survival_s=float(np.mean([float(r["sim_s"]) for r in selected])),
                           total_laps=sum(float(r["laps"]) for r in selected),
                           overtakes=sum(int(r["overtakes"]) for r in selected),
                           cycle_mean_ms=float(np.mean(latencies)),
                           cycle_p95_ms=float(np.percentile(latencies, 95)),
                           cycle_p99_ms=float(np.percentile(latencies, 99)),
                           cycle_max_ms=float(np.max(latencies)),
                           miqp_mean_ms=float(np.mean(high_latencies)),
                           miqp_p95_ms=float(np.percentile(high_latencies, 95)),
                           cycles=len(latencies), deadline_overruns=int(np.sum(latencies > 100)),
                           repairs=repairs, repair_successes=sum(int(r["repair_successes"]) for r in selected),
                           repair_amortized_ms=repair_ms / len(latencies),
                           repair_per_attempt_ms=repair_ms / repairs if repairs else 0.0,
                           fallbacks=sum(int(r["fallbacks"]) for r in selected),
                           unrecoverable=sum(int(r["unrecoverable"]) for r in selected)))
    paired = []
    for left, right in (("B1", "B0"), ("B1", "B2")):
        if left not in manifest["methods"] or right not in manifest["methods"]:
            continue
        by_method = {m: {(r["track"], r["pace"], r["seed"]): r["success"] == "True" for r in rows if r["method"] == m}
                     for m in (left, right)}
        a, b = by_method[left], by_method[right]
        paired.append(dict(comparison=f"{left} vs {right}", left_only=sum(a[k] and not b[k] for k in a),
                           right_only=sum(b[k] and not a[k] for k in a),
                           both=sum(a[k] and b[k] for k in a), neither=sum(not a[k] and not b[k] for k in a)))
    (output / "paired.json").write_text(json.dumps(paired, indent=2))
    (output / "totals.json").write_text(json.dumps(totals, indent=2))
    with (output / "totals.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(totals[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(totals)
    tracks = manifest["tracks"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for idx, method in enumerate(manifest["methods"]):
        for ax, metric, label in zip(axes, ["success", "sim_s", "overtakes"],
                                     [f"{manifest['max_time']:g} s completion rate", "Mean survival [s]", "Mean overtakes / episode"]):
            values = []
            for track in tracks:
                selected = [r for r in rows if r["track"] == track and r["method"] == method]
                values.append(np.mean([r[metric] == "True" if metric == "success" else float(r[metric]) for r in selected]))
            ax.bar(np.arange(len(tracks)) + (idx - 1) * 0.24, values, width=0.24, color=COLORS[method], label=method)
            ax.set_xticks(range(len(tracks)), tracks)
            ax.set_ylabel(label)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    fig.suptitle("SOfAR tracks — new, uncalibrated dynamic bicycle controller")
    fig.savefig(output / "comparison.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for method in manifest["methods"]:
        data = np.sort(np.concatenate(timing[method]))
        axes[0].plot(data, np.linspace(0, 1, len(data)), color=COLORS[method], label=method)
    axes[0].set_xlabel("Controller compute latency [ms], JIT excluded")
    axes[0].set_ylabel("Empirical cumulative probability")
    axes[0].set_xlim(0, max(r["cycle_max_ms"] for r in totals) * 1.05)
    axes[0].legend()
    axes[1].bar([r["method"] for r in totals], [r["repair_amortized_ms"] for r in totals],
                color=[COLORS[r["method"]] for r in totals])
    axes[1].set_ylabel("Local QP repair time / executed cycle [ms]")
    axes[1].set_title("Excludes batch barrier-cost overhead")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2)
    fig.savefig(output / "latency.png", dpi=180)
    plt.close(fig)
    candidates = [r for r in rows if r["track"] == "LS" and r["method"] == "B1"]
    representative = sorted(candidates, key=lambda r: (float(r["pace"]), int(r["seed"])))[0] if candidates else rows[0]
    track = Track.load(representative["track"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax in axes:
        ax.plot(track.inner[:, 0], track.inner[:, 1], color="#777777", linewidth=0.7)
        ax.plot(track.outer[:, 0], track.outer[:, 1], color="#777777", linewidth=0.7)
        ax.plot(track.geometry[:, 1], track.geometry[:, 2], color="#aaaaaa", linestyle="--", linewidth=0.6)
        ax.set_aspect("equal")
        ax.set_xlabel("World x [m]")
        ax.set_ylabel("World y [m]")
    for method in manifest["methods"]:
        p = source / f"trajectory_{representative['track']}_{float(representative['pace']):g}_{representative['seed']}_{method}.npz"
        samples = np.load(p)["samples"]
        xy = np.array([track.to_world(float(r[1]), float(r[2])) for r in samples])
        axes[0].plot(xy[:, 0], xy[:, 1], color=COLORS[method], linewidth=0.9, label=method)
        axes[1].plot(xy[:, 0], xy[:, 1], color=COLORS[method], linewidth=1.1)
    start = track.to_world(float(track.geometry[20, 0]), 0.0)
    axes[1].set_xlim(start[0] - 100, start[0] + 400)
    axes[1].set_ylim(start[1] - 150, start[1] + 350)
    axes[0].legend()
    fig.suptitle(f"Pre-selected episode: {track.name}, pace {representative['pace']}, seed {representative['seed']}")
    fig.savefig(output / "trajectory.png", dpi=180)
    plt.close(fig)
    text = ["# SOfAR 风格闭环实验：实际运行结果", "", "## 范围与计时边界", "",
            f"实际完成 {len(rows)} 次；每次上限 {manifest['max_time']} s。源码 SHA-256：`{manifest['source_sha256']}`。",
            "这不是作者 acados-MPC / RL 的复现。动力学未标定，速度参考有额外的 39 m/s 上限及提前制动，对手 pace 是原始 raceline 速度倍率。",
            "高 pace 档的对手在部分赛道/路段可能快于自车目标速度上限，对这些档次不能预设必能超车；完成率也不等于超车性能。",
            "计时包含控制器规划、采样、筛选与 shield，不包含仿真真值检测/绘图；JIT 已排除。计算延迟没有写回动力学；不是硬实时或严格 CBF 安全证明。",
            "", "## 汇总（所有失败均保留）", "",
            f"|方法|{manifest['max_time']:g} s 完成|碰撞|越界|平均存活 s|总圈数|超车|控制均值 ms|P95 ms|P99 ms|QP 修复均摊 ms|",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in totals:
        text.append(f"|{r['method']}|{r['completions']}/{r['n']}|{r['collisions']}|{r['boundary_failures']}|"
                    f"{r['mean_survival_s']:.1f}|{r['total_laps']:.2f}|{r['overtakes']}|{r['cycle_mean_ms']:.2f}|"
                    f"{r['cycle_p95_ms']:.2f}|{r['cycle_p99_ms']:.2f}|{r['repair_amortized_ms']:.3f}|")
    text += ["", "## 分赛道（不以总体均值掩盖失败）", "",
             "|赛道|方法|完成|碰撞终止|越界终止|平均存活 s|超车总数|", "|---|---|---:|---:|---:|---:|---:|"]
    for name in tracks:
        for method in manifest["methods"]:
            group = [r for r in rows if r["track"] == name and r["method"] == method]
            text.append(f"|{name}|{method}|{sum(r['success'] == 'True' for r in group)}/{len(group)}|" +
                        f"{sum(r['failure'] == 'collision' for r in group)}|" +
                        f"{sum(r['failure'] == 'track_boundary_violation' for r in group)}|" +
                        f"{np.mean([float(r['sim_s']) for r in group]):.1f}|{sum(int(r['overtakes']) for r in group)}|")
    lookup = {r["method"]: r for r in totals}
    text += ["", "## 本次观察（不是成功预设）", ""]
    if "B1" in lookup and "B0" in lookup:
        full, baseline = lookup["B1"], lookup["B0"]
        better = full["completions"] / full["n"] > baseline["completions"] / baseline["n"]
        text.append(f"完整方法 B1 完成 {full['completions']}/{full['n']}，基线 B0 完成 {baseline['completions']}/{baseline['n']}；" +
                    ("本批总体完成率更高，但尚不能证明普遍收益。" if better else "未观察到整体安全完成率优于基线，不能宣称 APF 方案已经有效。"))
    if "B1" in lookup and "B2" in lookup:
        shielded, unshielded = lookup["B1"], lookup["B2"]
        text.append(f"B1 / B2 完成数为 {shielded['completions']}/{shielded['n']} / {unshielded['completions']}/{unshielded['n']}，" +
                    f"末态物理接触为 {shielded['physical_end_contacts']} / {unshielded['physical_end_contacts']} 次。" +
                    "这是屏障代价加局部修复的组合比较；只有开销小不等于已验证安全收益，不能把原理预期当作本批结论。")
    text += ["动力学和跟踪未标定，失败终止原因、末态与不可恢复周期只能用于定位问题，不能据此断言单一失效原因。下一步优先验证无障碍跟踪和车辆参数，而不是继续增加方法组。"]
    text += ["", "延迟来自全部实际周期 pooled 数据，不是 episode P95 的平均。失败方法执行的周期较少、轨迹与状态不同，不能把两组耗时差直接归因为 shield 成本。",
             "碰撞/越界列为首要终止原因（同时违反时越界优先）。对最终车体独立 SAT 复核的物理接触次数：" +
             "；".join(f"{r['method']} {r['physical_end_contacts']} 次" for r in totals) + "。",
             "", "## Shield 与保底状态", "",
             "|方法|修复尝试|修复通过|回退|不可恢复周期|超过 100 ms 周期|", "|---|---:|---:|---:|---:|---:|"]
    for r in totals:
        text.append(f"|{r['method']}|{r['repairs']}|{r['repair_successes']}|{r['fallbacks']}|{r['unrecoverable']}|{r['deadline_overruns']} / {r['cycles']}|")
    text += ["", "高层 MIQP 规划调用 pooled 均值 / P95（ms，含无活跃对手时的快速返回）：" +
             "；".join(f"{r['method']} {r['miqp_mean_ms']:.2f} / {r['miqp_p95_ms']:.2f}" for r in totals) + "。",
             "最大控制周期（ms）：" + "；".join(f"{r['method']} {r['cycle_max_ms']:.2f}" for r in totals) + "。"]
    text += ["", "## 同初始状态配对完成计数", "", "|比较|仅左组完成|仅右组完成|均完成|均失败|", "|---|---:|---:|---:|---:|"]
    for pair in paired:
        text.append(f"|{pair['comparison']}|{pair['left_only']}|{pair['right_only']}|{pair['both']}|{pair['neither']}|")
    text += ["", "QP 均摊时间只计事件触发局部修复，未单独测量 rollout 屏障代价开销。`unrecoverable` 表示没找到预测可接受的序列，不等于当步已经碰撞。",
             "", "## 解释边界", "",
             "- B0/B1 比较偏置候选方案，包含 APF 与原子候选的共同作用，不能归因于单独一个 APF 分量。",
             "- B1/B2 比较软屏障代价与事件触发修复的组合，不是仅 QP 的单独消融。",
             "- 每个 track/pace 仅五个初始状态，结果用于探索性筛选，未作显著性或真实车辆泛化声明。",
             "- 三组同 seed、同对手策略；回收时间随自车进度变化，因此回收后不保证同一绝对对手轨迹。",
             "- 完成率不能单独代表赛车速度；应与所有 episode 的圈数、超车和存活时间一起阅读。",
             "- 开发试跑及调参失败保留在独立目录，不计入冻结源码的正式样本。",
             "- 真值几何在 0.1 s 执行采样点判断，不是连续时间扫掠碰撞证明。",
             "- 固定种子与配置不等于跨硬件位级复现；Gurobi 墙钟时限可能影响边缘情形的可行解。",
             "", "## 图", "", "![赛道对照](comparison.png)", "", "![逐周期计时](latency.png)", "", "![预先固定选例](trajectory.png)",
             "", "复现：参见 `experiments/README.md`，逐次原始 CSV 与配置 manifest 一并交付。数据源是本地产生的 CSV/NPZ，不需要数据库；可复现聚合脚本为 `experiments/report.py`，不是 SQL 查询。"]
    (output / "REPORT.md").write_text("\n".join(text) + "\n")
    print(json.dumps(totals, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report(args.input, args.output)


if __name__ == "__main__":
    main()
