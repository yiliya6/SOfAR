# MIQP / Directional-APF / Biased-MPPI headless experiments

独立实验入口，直接读取本仓库作者公开的四条 raceline 和 boundary。**不是作者 acados-MPC / RL 的完整复现，也不是已获安全证明的 CBF 控制器。**

## 运行

在仓库根目录使用 Python 3.10–3.12 的独立虚拟环境安装，避免与原版的老 numpy/scipy 依赖混装（本次运行 Python 3.10）：

```sh
python -m venv .venv-mppi
.venv-mppi/bin/python -m pip install -r experiments/requirements.txt
# 需要可用的 Gurobi 许可证；测试规模适用于 restricted non-production license。
.venv-mppi/bin/python -m pytest -q experiments
.venv-mppi/bin/python -m ruff check experiments
.venv-mppi/bin/python -m mypy experiments --follow-imports skip --ignore-missing-imports
.venv-mppi/bin/python -m experiments.run --suite development --output batch_results/development_v2 --trajectories
.venv-mppi/bin/python -m experiments.run --suite final --output batch_results/final_v2 --trajectories
.venv-mppi/bin/python -m experiments.report --input batch_results/final_v2 --output batch_results/report
```

development 为 LS × {0.3,0.65} × seeds 101–105 × 三方法 = 30 次；final 为 {KS,IMS,LS,Monza} × {0.1,0.3,0.5,0.65,0.8} × seeds 201–205 × 三方法 = 300 次。每次最多 200 s，碰撞、车体跨边界或非有限状态即失败，失败不得删去。目录存在时可断点继续；配置、源码哈希或依赖版本不一致时拒绝混入新结果。

## 算法和公平性

| 方法 | 高层 | 候选，共 128 条 | 屏障代价 / 局部修复 |
|---|---|---|---|
| B0 | MIQP | Gaussian / warm start，使用同一 MIQP 横向目标 | 开启 |
| B1 | MIQP | 64 Gaussian + 63 APF-biased Gaussian + 1 APF atom | 开启 |
| B2 | MIQP | 与 B1 相同 | 关闭 |

三个方法使用相同初始 seed、对手初始化 / 移动 / 前方回收策略、动态模型、样本总数、几何验收和减速回退。回收时间随自车进度而变化，所以回收之后的对手绝对轨迹不是跨方法完全相同。

- MIQP：每 0.5 s 决策；18 × 0.25 s 横向规划，障碍左右二元变量、赛道边界、连续性、重罚松弛；25 ms 求解时限。只有预测到纵向交会时才输出绕行侧，失败时保留已有模式。横向目标每周期变化不超过 0.3 m，避免瞬时换侧。
- 动力学：状态 `[s,n,heading_error,vx,vy,yaw_rate,steer]`，输入 `[acceleration,steering_rate]`，动态 bicycle、饱和线性侧偏力、0.1 s 控制步、两个 0.05 s RK2 子步。参数为未标定赛车代理，不是 AV-21 精确模型。
- APF：MIQP 目标吸引、椭圆径向排斥、边界排斥；切向方向严格取 MIQP 给定的符号，并映射为加速度和转向速率偏置。APF 不自行选左/右。其输出是候选，不直接覆盖 MPPI。
- 下层为 **MPPI-style Gibbs weighted shooting**：2 s horizon、温度 45、固定混合提案，有限候选按 `exp(-cost/45)` 加权。定义的是相对于提案混合测度的工程优化；未实现保持纯 Gaussian 目标不变的混合密度重要性校正，原子候选也没有伪装成 Gaussian KL。
- 加权均值重新通过完整 20 步非线性几何验收；B0/B1 另验收前 6 步 DCBF residual `h_next-0.5*h >= 0`。触发修复时，仅优化两个 2-step 控制块，共 4 变量、最多两次 OSQP 顺序凸化；不对每条 rollout 求 QP。
- 修复失败则选择已重新验收的样本；均无可接受序列时执行共同的模式保持减速回退，并明确记录 `unrecoverable`。此回退不是安全保证，实际碰撞仍独立记失败。
- 真值碰撞为世界坐标有向车体矩形 SAT；真值越界为车体角点和边与真实边界多边形检查。预测检查为 Frenet 保守代理，不构成一般曲率下的形式化包络证明。

速度参考使用 `min(0.8*published_raceline_speed,39m/s,sqrt(16/abs(curvature)))`，再做 7 m/s² 的循环提前制动包络；仍有跟车减速。对手 pace 只缩放原始 raceline velocity，**不是即时对手/自车速度比**。这与作者原版速度控制不同。

## 输出与局限

- `episodes.csv`：逐次真实结果，200 s 完成、终止原因、存活时间、圈数、超车数、每周期平均/P95/P99、MIQP、QP 修复、不可恢复、100 ms 超时数。
- `trajectory_*.npz`：每个实际周期状态、输入、MIQP 侧、对手坐标、逐周期延迟及各次 MIQP 时间。
- `manifest.json`：配置、源码 SHA-256、Python/依赖版本、CPU 数、模型及计时边界。
- `summary.json`：快速按赛道/方法汇总；其中 `mean_cycle_ms` 是 episode 均值的平均。报告会另从逐周期记录计算 pooled 均值和分位数，避免混淆两者。

计算时间只包含控制器（规划、采样、筛选、shield），不包含绘图、真值碰撞判定、操作系统硬实时调度。JIT 编译排除在计时之外。计算延迟未写回植物动力学；100 ms 超时只是记录，不是延迟注入实验。`repair_total_ms` 仅量化事件触发 QP 修复，不单独隔离 batch 内屏障代价的额外耗时。短前缀 DCBF 是工程过滤而非高相对阶 CBF/HOCBF 不变性证明。

CSV 的 `failure` 是首要终止原因：同时越界和碰撞时，runner 先记越界。报告另外对保存的最终世界坐标车体独立复核碰撞，避免把“零碰撞终止”误当成“完全没有物理接触”。

断点恢复同时校验源码、依赖和赛道文件哈希（包括对手初始化/回收使用的 LS 数据）；请求轨迹时会重跑丢失或损坏 NPZ 的 episode，并原子替换对应 CSV 行。自定义时长必须为至少 0.1 s 的整数控制步。报告须在原冻结源码/赛道版本下生成。`development_v1` 和中止的 `final_v1` 含已确认的预测/目标选择错误，只作为历史调试记录，不作为有效方法结论。

复现固定随机种子、初始条件和配置，不承诺跨硬件的位级一致性：Gurobi 的墙钟时限可能导致边缘情形使用不同可行解。真值几何只在 0.1 s 执行点检查，没有连续时间扫掠碰撞证明。

本实现的开发试跑可能失败；正式结果必须完整保留，不能以单次成功、计算均值或原版论文结果替代本实现实际统计。
