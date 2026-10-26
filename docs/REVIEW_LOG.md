# 审查记录（REVIEW_LOG）

## 1. 自审（2026-10-20，计划 v0.1 → v0.2）

逐条把 `REPRODUCTION_PLAN.md` 对照 `PAPER_SPEC.md` 与 mjlab 1.6.0 源码复核。发现并已修正：

| # | 发现 | 处理 |
|---|---|---|
| S1 | D4 原用相邻预测状态的有限差分算刚体速度，只能做到 5% 容差，且与仿真的 `cvel` 定义不一致 | 改为由 qvel 沿运动树传播的速度 FK，可对 MuJoCo 精确对拍；有限差分降为备选 |
| S2 | 论文写 "per-episode sim context"，mjlab 跟踪任务的 DR 是 `startup` 一次性采样 | D8 改为 `mode="reset"` 按回合重采样；replay buffer 按步存 ξ；保留 `startup` 为备选 |
| S3 | 原 6.4 在训练中每 4000 迭代跑全量 benchmark，80 次 × 15–30 min 会吞掉大部分训练时间；论文也不把评测计入训练 | 改为只存检查点，评测离线或在空闲卡上跑 |
| S4 | 2.6 的状态回写测试要求"下一步与记录轨迹一致"，但求解器 warmstart 会带来微小差异 | 容差改为归一化误差 ≤ 1e-3 |
| S5 | 观测头的基座角速度直接取 qvel[3:6]，忽略了 IMU site 相对骨盆可能有固定旋转 | 2.9 明确要按 site 姿态变换，并由对拍测试覆盖 |
| S6 | 1.3 的动作库若按帧存 14 刚体位姿/速度，BONES-SEED 全量需约 35 GB | 只存 71 维 (qpos, qvel)，目标帧与奖励所需刚体量在线 FK |
| S7 | critic 的状态输入未说明是否规范化 | D10 明确用航向规范化表示 |
| S8 | 终止头输入描述含糊 | D4 明确输入为预测根高度/投影重力与参考根高度/参考投影重力 |
| S9 | R5 描述的是有限差分风险，S1 后不再成立 | 改为"解析奖励头实现错误"的风险，以对拍测试为 gate |

复核通过、无需修改的项：
- SPEC §5 奖励表、§7 DR 表、§11 超参表与论文 Table A.2/A.3/A.5 逐项一致。
- 维度推导：s 71、ξ 33、o 93、历史 930、目标 680、z 32。
- D12 的时间自洽性论证（仿真 3.6 h + 世界模型 2–3 h + 策略 2–3 h ≈ 论文单卡 10 h）。
- 显存估算与论文"H=128 在 80 GB 上刚好放下"自洽。
- 终止用骨盆而非 mjlab 的锚点 torso_link，与论文 "robot root (pelvis)" 一致。
- 去掉 push、去掉末端执行器终止、去掉自碰撞惩罚、关节限位 −2，均与论文一致。

## 2. 对抗审查

两个独立审查 agent，分别从"论文保真度"和"工程可行性（mjlab 底座）"两个角度攻击计划。每条发现我都先回到 mjlab 源码或论文原文核实，再决定处理。

### 2.1 工程可行性审查（计划 v0.2 → v0.3）

| # | 严重度 | 发现 | 核实 | 处理 |
|---|---|---|---|---|
| E1 | critical | `dr.joint_default_pos` 只写 `qpos0`，动作偏置与 `joint_pos_rel` 读的是实体初始化时复制的 `default_joint_pos`，所以 D8 的"同时偏移观测与设定点"不成立 | 已核实 `dr/joint.py:178-202`、`actions.py:218-219`、`observations.py:51-61`。另核实 `encoder_bias` 同时作用于观测（`data.py:390-392`）与执行（`actions.py:221-223` 目标减去 bias） | D8 改用 `encoder_bias`；2.3/2.7 同步修改 |
| E2 | critical | 计划固定的 `mjlab==1.6.0`（PyPI）不是我调研的代码：HEAD 的 changelog 标 "Upcoming version"，v1.6.0 发布于 2026-08-08（tag `0fb8a68`），mujoco-warp 3.11 → 3.16、cu126 extra、csv_to_npz 末帧修复均在 HEAD | 已核实 `changelog.rst:5, 102`，`git ls-remote` 确认 tag | 改为 git 依赖固定到 commit 8bb2bc9；文首与 §2.2/2.3 改写；R9 改写 |
| E3 | major | reset 模式的 COM 与 qpos0 DR 会在任一重置后触发全局 `set_const` 重算，4096 环境下等于每步一次 | 已核实 `dr/body.py:328`、`event_manager.py:332-337`、`sim.py:479-490` | 摩擦与 encoder_bias 用 reset 模式（不触发重算），COM 用 startup；0.3 吞吐基准改为用真实 DR 配置；R11 |
| E4 | major | 显存估算偏低：按 autograd 实际保存量激活约 16.4 GB，replay 139 维 2.3 GB，峰值 24–26 GB；H=64 在 4096 批下放不下 | 按 autograd 保存量重新推算 | §9 显存表重写；§8 H=64 注释 |
| E5 | major | 奖励对拍 1e-4 在 `env.step` 路径下不可达（运动学晚一个子步），且 D4 的 ≤1% 与 2.8 的 ≤2% 矛盾 | 已核实 `manager_based_rl_env.py:395-410`；奖励函数只读命令项属性 | 2.8 改为先 `sim.forward()` 再直接调用 mjlab 奖励函数，容差 1e-5；采集器在 step 后自行重算 r |
| E6 | major | 观测历史是 term-major 布局；组级 `history_length` 会覆盖组内所有项，目标项会被堆成 6800 维 | 已核实 `observation_manager.py:63-65, 380-383, 461-463` | D20 重写：目标项独立观测组；想象侧复用 `CircularBuffer`；R12 |
| E7 | major | 想象 rollout 每次更新约 2 万次 kernel launch，未做 compile/CUDA graph 就与 graph 回放的仿真比速度不公平 | 合理 | 新增 3.6 "启动开销控制（必做）"；时间估算加 50% 余量；R14 |
| E8 | major | mjlab 的 torch 索引与 sources 不会传递给下游；cu128 路线需实测 warp 在 570 上可用；排期风险缺项 | 已核实 `pyproject.toml:112-137` | §2.2/2.3 补索引声明与验证命令；R13、R17 |
| E9 | major | FK 测试 1e-5 在 TF32 下不可达；速度量级大需 rtol；状态回写容差应按分位数；有限差分需 float64 | 合理 | 2.2、2.6、4.3 容差改写；R15 |
| E10 | major | `csv_to_npz.py` 逐帧 slerp 与逐帧 forward 太慢，不适合 47.6M 帧；BONES-SEED 关节顺序/四元数顺序/fps 未验证就假定一致 | 已核实 `csv_to_npz.py:108-115, 265-281, 312-329` | 1.1 改为批量实现；1.4 设为硬门；R16 |
| E11 | minor | 向 mjlab registry 注册需要 rsl_rl 的 `rl_cfg` | 已核实 `registry.py:22-28` | 改为 `make_env()` 直接构造 |
| E12 | minor | 重复实现 rotations / 历史缓冲 / projected_gravity / seeding | 合理 | 骨架改为复用 `mjlab.utils.lab_api.math`、`CircularBuffer`、`seed_rng` |
| E13 | minor | IMU site 只有位置没有旋转，自审 S5 的变换说法错误 | 已核实 `g1.xml:69` | 2.9 更正 |
| E14 | minor | 锚点相对系只对齐 xy + yaw；奖励的"根"是 torso_link 而非骨盆 | 已核实 `commands.py:379-407` | D7 重写；终止刚体进配置，默认骨盆（按论文） |
| E15 | minor | 回合起点 a_{−1} 应为 0 | 已核实 `action_manager.py:140-146, 166-169` | D9 补充 |
| E16 | minor | `auto_reset=False` 每次重置多一次 forward；recorder pre-reset 钩子更便宜 | 已核实 `manager_based_rl_env.py:371-373, 470-471` | Phase 2 注意事项改写 |
| E17 | minor | clip 末尾 timeout 的时序与重置路径 `+1` 约定 | 已核实 `:452, 486-488` | 2.4 补充 |
| E18 | minor | 多卡需自行 `init_process_group`；学习率缩放应可配置；每 rank 一份 13.5 GB 锁页库太大 | 合理 | 6.3 改写 |
| E19 | minor | Phase 2 估时 3–4 天、Phase 4 无调参余量不现实；风险表缺项 | 合理 | §9 排期改为 8–10 周；风险表 R11–R17 |

审查方确认正确的项（不再复述）：安装要求 580+、torchrunx 启动、动作缩放数值、14 跟踪刚体、重置噪声、sim 时序、命令 +1 约定、DR 引擎支持 env_ids 与 reset 模式、原地写 qpos/qvel 后 forward 在 CUDA graph 下可行、CircularBuffer 回填、噪声在历史之前施加、per-world 模型字段可读回、csv 列序与四元数序、47.6M 帧与 13.5 GB 算术、D12 时间自洽推导。

### 2.2 论文保真度审查（计划 v0.3 → v0.4）

| # | 严重度 | 发现 | 核实 | 处理 |
|---|---|---|---|---|
| F1 | major | D12 "每迭代 0.4 次更新"与 Algorithm A.1 "Both ω and θ update every iteration" 冲突；时间自洽性论证依赖图读数据（±2×），不能排除其他解读 | 论文原文核实；同意论证不充分 | D12 重写：调度比例（每 2.5 个仿真步 1 次更新）为确定部分；总长度 (A)/(B)/(C) 三种读法进配置、默认 (A)；学习率调度按比例定义；用正文的 1.2 h / 3 h 锚点在 e2 反向校准 |
| F2 | major | 观测中的 a_{t−1} 应为策略的干净动作；噪声"serves only the model fit"，不应进入策略输入分布；D20 的历史重建会混入噪声 | 论文 §3.1 记号核实 | D9 重写：新增 `NoisyJointPositionAction`，干净动作入观测与平滑项，噪声只在执行时加；buffer 同时存 a 与 a_exec；2.3、§3.1 图、§5 同步 |
| F3 | major | G_max 按论文是单一常量（≈10），计划改成逐 bin 未标注；10 s 超时截断的回报-to-go 会把临近超时的步系统性判为"难" | 同意 | D14 重写：G_max = 10；超时用 critic 自举；clip 末尾不自举；备选进配置 |
| F4 | major | critic 输入用航向规范化状态且 g_t 不含 xy/yaw，看不到世界系根部漂移，V(ŝ_H) 对漂移梯度恒零，丢失 H 之外的根部回报信号 | 同意，这是实质性缺陷 | D10 重写：critic 增加 e_root（参考航向系根位置误差 3 + yaw 误差 1）；Critic 接口签名更新 |
| F5 | major | 关键词若裸词子串匹配会误杀（sit→position，box→boxing，ball→ballet）；未定义匹配方式；只用最终 ±1% 门槛不够 | 核实 SONIC 脚本为小写子串匹配 | D16 重写：沿用子串匹配但新增词一律下划线定界；按 Table A.7 三个数字分别设 ±0.5% 门槛并输出逐词命中统计 |
| F6 | minor | §3.1 图中 L_WM 含 reward + obs 项，与 D4 解析头矛盾 | 同意 | 图改为 state + BCE(term)，注明解析头无损失 |
| F7 | minor | 状态损失取各维均值，相对 BCE 的权重比论文小约 70× | 同意 | D3 改为按维求和；BCE 权重进配置 |
| F8 | minor | 未写明 ĉ_t 对 d̂ 保留梯度、k 步拟合不截断梯度 | 同意 | D10 增加梯度流清单（5 条） |
| F9 | minor | 评测漏项：重置噪声需关闭、初始状态含速度精确等于首帧；5.3 的 0.82 用训练回报还是评测回报；训练回报分母；归一化 off-by-one | 同意 | 5.1、5.3 重写 |
| F10 | minor | SPEC 把"每 GPU 4096 环境"误列为未指明；U19 可由 F.1 + §4 推出 | 同意 | SPEC §11、U13、U19 修正 |
| F11 | minor | 多处图读值被当作论文给定值（Fig. 3b/c 数值、Fig. 5a 的 4k≈11×、"≈10 h"、M3 的 0.85） | 同意 | SPEC §15 与计划 M3、§9、附录 B 标注"图读" |
| F12 | minor | D4 与 2.8 的速度 FK 精度声明矛盾 | 工程审查轮已把 2.8 改为 1e-5 | 已解决，无需再改 |
| F13 | minor | §3.2 "复用 mjlab 奖励函数"与 §5 自写共用实现矛盾 | 同意 | §3.2 表重写：mjlab 奖励/终止只作对拍基准 |
| F14 | minor | replay 估算漏 ξ；策略更新 FLOP 中冻结的动力学网络只需输入梯度（×2×2） | 同意 | §9 replay 改为 169 维 2.8 GB、峰值 25–27 GB；附录 B 改为 11.7 TFLOP |
| F15 | minor | 推力测试缺静态 T-pose 参考 clip、推力施加方式、稳定判据 | 同意 | 5.2 补全（合成 20 s T-pose clip；`push_by_setting_velocity` 方式；自定稳定判据并标注论文未给） |
| F16 | minor | critic 的 Adam、学习率、范数裁剪 1.0、minibatch 总体未写 | 同意 | 4.4 补全 |
| F17 | minor | 终止步的 s_{T+1} 在 [env, time] 布局下没有存储位置 | 同意 | replay 在重置前多写一行 terminal 行；`add_terminal()` 接口；采样器排除 terminal 行作为起点 |
| F18 | minor | 8 步误差应在新采集的 rollout 上在线测，不在训练批上 | 同意 | D3、3.4 修正 |
| F19 | minor | SPEC 把 TF32 外推到策略网络 | 同意 | SPEC §11 修正 |
| F20 | minor | SPEC 对蒸馏基线摔倒条件转录不完整（"combining translation with a turn"） | 同意 | SPEC §16 修正 |

审查方确认正确的项（不再复述）：Table A.1–A.5 全部数值与公式、Eq. 1–3 与存活权重、终止阈值、ξ 拆分、网络表、D13 多卡读法、参数量与激活估算、Table A.7/A.8、可行性过滤阈值、F.1 评测条件、Appendix G 全部数值、噪声与确定性策略的处理。

## 3. 审查后仍开放的事项

- D12 的总长度解读需要 e2 的实测校准，计划已把三种读法进配置。
- D5 的目标帧内容、D1 的规范化、D3 的名义尺度、D14 的 EMA 系数都是论文未指明处的自选值，首个配置是否达标要靠 Phase 4 的迭代；全部可配置，EXPERIMENTS.md 记录每次变更。
