# SuperMimic 复现计划（REPRODUCTION_PLAN）

**状态：v0.4 于 2026-10-21 定稿并按此实施。本文件保留为实施骨架与设计决策的依据；实施中偏离计划的决定与实测数据见 `docs/EXPERIMENTS.md`，不回改本文件。**

配套文件：
- `docs/PAPER_SPEC.md`：论文每个数值、公式、协议的逐条转录，含"未指明"清单 U1–U20。本计划引用时写作 SPEC §n / Un。
- `docs/REVIEW_LOG.md`：自审清单与对抗审查记录及其处理。
- 论文原文：https://arxiv.org/abs/2610.28480（PDF 不随仓库分发）。

mjlab 引用格式：`mjlab/<path>:<line>`，对应 mjlab **main 分支 commit 8bb2bc9**（2026-10-18）源码树 `src/mjlab/` 下的路径。注意：该提交的 `pyproject.toml` 版本号字段写的是 1.6.0，但 changelog 标明它是"Upcoming version (not yet released)"，正式发布的 v1.6.0 是 2026-08-08 的 tag `0fb8a68`，两者在 mujoco-warp 版本（3.11 → 3.16）、`cu126` extra、`csv_to_npz` 末帧修复等处不同。本计划一律**固定到 commit 8bb2bc9**，不用 PyPI 的 1.6.0。

---

## 0. 一页总览

**目标**：在 1×RTX 5090（机器 A）与 4×RTX 5090（机器 B）上，按论文 Algorithm A.1 从零实现 SuperMimic（学习的可微世界模型 + BPTT 短视野 actor-critic），先在 LAFAN1 40 条 clip 上复现 Fig. 4 的量级结果，再在 BONES-SEED 全量（129,514 条）上复现 Fig. 3 / Table A.8 的量级结果。真机部署不在范围内（无 G1）。

**底座**：mjlab（MuJoCo Warp 的 Isaac Lab 风格框架，共同作者 Zakka 维护），固定到 main 分支 commit 8bb2bc9，作为 git 依赖，不 fork。复用其 G1 资产、仿真封装、manager 式环境、动作跟踪任务的奖励/终止函数与 CSV 转换逻辑。重写：多 clip 动作库、未来 K 帧目标、基于回报的重置采样、完整状态读写、世界模型、BPTT 训练器、评测。不使用 rsl_rl。

**里程碑与验收**：

| 里程碑 | 内容 | 验收标准 | 累计估时 |
|---|---|---|---|
| M0 | 环境就绪 | mjlab G1 跟踪任务 PPO 冒烟跑通；4096 环境仿真吞吐实测 | 第 1 周 |
| M1 | 数据与环境层 | LAFAN1 多 clip 库；自定义环境通过全部单元测试（FK、奖励、观测头、状态回写） | 第 2–3 周 |
| M2 | 世界模型离线验证 | 8 步归一化误差稳定；256 步滚动不发散；编译后速度 ≥ 5× 仿真 | 第 4 周 |
| M3 | LAFAN1 40 clip 训练 | 单卡 5090 上训练回报 ≥ 0.82（论文 Fig. 4b/c 的判据）；benchmark 评测回报 ≥ 0.82（论文 Fig. 4a 正文给出的数；图上最终约 0.85，图读仅作参考） | 第 5–7 周 |
| M4 | 评测与消融 | benchmark 脚本、推力测试；H ∈ {16, 32} × {有/无 critic} 子集消融 | 第 8 周 |
| M5 | BONES-SEED 全量 | 过滤得 ≈129.5k 条；4×5090 DDP 训练；失败数随训练下降 | 第 9–10 周 |
| M6（可选） | 速度接口、ONNX 导出、PPO-Sim 基线 | 按需 | 之后 |

估时假设一人全职，含调试；估时的依据见第 9 节。

---

## 1. 范围与不做的事

**做**：
- 动作跟踪接口的完整训练循环（SPEC §12），LAFAN1 与 BONES-SEED 两档数据。
- 论文 Fig. 3a/c、Fig. 4a/b/c、Fig. 5a、Table A.8 的对应实验（规模按算力缩放）。
- 单卡与多卡（数据并行）训练。

**不做 / 可选**：
- 真机部署（无 G1 硬件）。
- 与 SONIC 权重的直接对比：SONIC 权重在 HuggingFace `nvidia/GEAR-SONIC` 以 ONNX + TensorRT 形式发布，需要搭建其观测管线，列为 M6 可选项。
- 速度接口（SPEC §16）、PPO-WM 消融（Fig. 5b）、PPO-Sim 基线（Fig. 4a 的 PPO 曲线）：M6 可选项。

---

## 2. 硬件与软件环境

### 2.1 硬件（已实测，2026-10-20）

| | 机器 A | 机器 B |
|---|---|---|
| GPU | 1 × RTX 5090 32 GB | 4 × RTX 5090 32 GB，PCIe，无 NVLink |
| CPU / 内存 | 32 核 / 62 GB | 128 核 / 251 GB |
| 磁盘 | – | 1.2 TB 可用 |
| 驱动 | 580.178（CUDA 13.0） | 570.172（CUDA 12.8） |

开发期需要机器 A 的 5090 至少 ≈26 GB 可用显存（见第 9 节）。

**阻碍项 B1：机器 B 的驱动。** mjlab main 训练要求驱动 ≥ 580（默认 torch cu130 轮子；`cu126` extra 不含 Blackwell 内核），见 `mjlab/docs/source/installation.rst:17-18`。机器 B 的 570.172 不满足。两条路：
- 推荐：把机器 B 的驱动升级到 ≥ 580。
- 备选：在我们自己的 `pyproject.toml` 里声明 `https://download.pytorch.org/whl/cu128` 索引并固定 torch cu128 轮子。warp-lang 1.18 的轮子自带 CUDA 运行时，`installation.rst` 说 `cu126` extra 可在 525+ 驱动上用，暗示 warp 轮子是 CUDA 12 构建，所以 cu128 路线大概率可行，但必须实际验证：`python -c "import warp as wp; wp.init(); print(wp.get_cuda_driver_version(), wp.get_cuda_toolkit_version())"` 并编译一个 kernel、跑一次 `mujoco_warp` 的 G1 场景。只在升级驱动不可行时走此路。

### 2.2 软件栈

- Python 3.12，`uv` 管理，锁定依赖。
- `mjlab @ git+https://github.com/mujocolab/mjlab@8bb2bc9e187797689ec82c78c35019afd2fd0376`（带 `mujoco-warp~=3.16.0`、`mujoco~=3.16.0`、`warp-lang>=1.15.0`、`torch>=2.14.0`、`torchrunx`）。
- mjlab 自己的 `[tool.uv.index]` / `[tool.uv.sources]`（torch 走 pytorch-cu130 索引、warp-lang 走 pypi.nvidia.com）**不会传递**给下游项目，我们的 `pyproject.toml` 必须自己声明同样的索引与 sources。
- 我们自己的包 `supermimic`，依赖 mjlab，不依赖 rsl_rl 的训练逻辑。
- 日志：TensorBoard（必需）+ W&B（可选开关，默认关）。
- 开发在机器 A 交互调试；长实验在机器 B。

### 2.3 安装步骤（Phase 0 执行）

```bash
cd SuperMimic
uv init --python 3.12 --lib   # 已有 pyproject 时跳过
# 在 pyproject.toml 中加入（与 mjlab 相同的索引与 sources）：
#   [[tool.uv.index]] name="pytorch-cu130" url="https://download.pytorch.org/whl/cu130" explicit=true
#   [[tool.uv.index]] name="nvidia" url="https://pypi.nvidia.com/" explicit=true
#   [tool.uv.sources] torch={index="pytorch-cu130"}  warp-lang={index="nvidia"}
#   mjlab = { git = "https://github.com/mujocolab/mjlab", rev = "8bb2bc9e187797689ec82c78c35019afd2fd0376" }
uv add mjlab tensorboard tyro
uv add --dev pytest ruff pyright
uv run python -c "import mjlab, mujoco_warp, warp, torch; print(torch.cuda.get_device_name(0), torch.version.cuda)"
uv run list-envs                                 # 应列出 Mjlab-Tracking-Flat-Unitree-G1
```

---

## 3. 总体架构

### 3.1 数据流（对应论文 Fig. 2）

```
┌──────────────── (1) Collect ─────────────────────┐
│ mjlab ManagerBasedRlEnv (4096 envs, 50 Hz)      │
│   a_t = π_θ(o_hist, g)  (clean, enters obs/reward)│
│   sim executes a_t + ε_t (noise inside action term)│
│   per-step row: s_t(71), a_t(29), a_exec(29),     │
│     ξ(33), r_{t+1}, d_{t+1}, clip, frame, flags   │
│   extra terminal row (s_{T+1}) written pre-reset   │
└───────────────────┬─────────────────────────────┘
                    ▼
            ReplayBuffer [env, time] ring, 1000 steps/env
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
(2) Fit world model      (3) Improve policy
  sample k=8 windows       sample 4096 start states (s_0, ξ, clip, frame, hist)
  roll f_ω on a_exec       imagine H=32 steps: ō̂ = g(ŝ,a_prev,ξ) + η,
  L_WM = Σ_dims state        a = π(ō̂_hist, g), ŝ' = f_ω(ŝ,a,ξ),
    + BCE(term)              r̂ = r(ŝ,ŝ',a,g) (analytic FK), d̂ = d_ω(ŝ',g)
  (reward/obs heads        J = Σ γ^t ĉ_t r̂ + γ^H ĉ_H V(ŝ_H,g_H,ξ,e_root) → BPTT → θ
   analytic: no loss)      GAE targets (Eq.3) → fit V_ψ (Adam, 5 epochs × 4 minibatch)
```

### 3.2 与 mjlab 的分工

| 复用（不改） | 重写 / 新增 |
|---|---|
| G1 MJCF、执行器、默认姿态（`mjlab/asset_zoo/robots/unitree_g1/`） | 多 clip 动作库 `MotionLibrary` |
| `Simulation`、`WarpBridge` 零拷贝状态访问（`mjlab/sim/`） | 多 clip 命令项 + 未来 K 帧目标 |
| `ManagerBasedRlEnv` 与 manager 体系 | 基于回报的重置采样 |
| 域随机化函数 `mjlab/envs/mdp/dr/`（`geom_friction`、`encoder_bias`、`body_com_offset`） | 完整状态读/写 `StateIO` |
| `mjlab.utils.lab_api.math`、`CircularBuffer`、`seed_rng`、`mdp.projected_gravity` | 世界模型、BPTT 训练器、critic |
| 指标 `mdp/metrics.py`（MPKPE 等） | 可微位置 FK + 速度 FK（torch） |
| `csv_to_npz.py` 的列序 / 四元数约定（只作规范，不调用其逐帧实现） | 跟踪奖励与终止的 torch 实现（仿真与想象共用，对预测状态可算）。mjlab 的 `mdp/rewards.py`、`mdp/terminations.py` 只作**对拍基准** |
| `torchrunx` 启动方式（`mjlab/scripts/train.py:182-227`） | `NoisyJointPositionAction`（D9）、批量本地转换器（去 W&B）、benchmark 评测、推力测试、DDP 训练入口 |

### 3.3 关键张量维度

| 量 | 维度 | 说明 |
|---|---|---|
| s_t | 71 | qpos 36（根位置 3 + 根四元数 wxyz 4 + 关节 29）+ qvel 35（根线速度 3 + 根角速度 3 + 关节 29） |
| a_t | 29 | 归一化设定点 |
| ξ | 33 | 足部摩擦 1 + 关节标定偏移 29 + 躯干质心偏移 3 |
| o_t | 93 | SPEC §2 |
| 历史观测 | 930 | 10 × 93 |
| 每帧目标 | 68 | 见 D5 |
| g_t | 680 | 10 × 68 |
| z | 32 | tanh |
| e_root（仅 critic） | 4 | 参考航向系根位置误差 3 + yaw 误差 1（D10） |

网络参数量估算：动力学 ≈ 8.8M，动作编码器 ≈ 4.2M，解码器 ≈ 4.7M，critic ≈ 0.23M，合计 ≈ 18M。

---

## 4. 设计决策（对应 SPEC §17 的 U1–U20）

每条：决定 / 理由 / 备选 / 验证。全部可通过配置切换到备选。

**D1（U1）状态向量与规范化坐标系。**
- 决定：s_t = MuJoCo 原始 (qpos, qvel)，71 维，这是仿真器的 Markov 状态（位置执行器无内部状态；`qacc_warmstart` 影响忽略）。动力学网络的输入做航向规范化：去掉根部 xy 与 yaw，输入 = [根高度 1, 去 yaw 后的根旋转 6D 2 列 6, 航向系根线速度 3, 本体系根角速度 3, 关节位置 29, 关节速度 29] = 71。网络输出航向系增量，在网络外积分回世界系（可微）。
- 理由：动力学对 xy 平移与 yaw 旋转不变；显式去掉可大幅提高数据效率。SuperTrack 也用局部系表示。
- 备选：直接输入世界系 qpos/qvel（配置 `canonicalize=False`）。
- 验证：`test_state_io_roundtrip`（读状态 → 写回 → forward → 再读，逐元素相等）；规范化的可逆性测试。

**D2（U4）残差参数化与四元数积分。**
- 决定：f_ω 输出 Δ = [Δ根位置(航向系) 3, Δ根旋转向量 3, Δ根线速度 3, Δ根角速度 3, Δ关节位置 29, Δ关节速度 29] = 70；ŝ_{t+1} = integrate(ŝ_t, Δ)，根四元数用 q ⊗ exp(Δrotvec) 更新并归一化。输出按名义尺度（D3）缩放。
- 理由：残差 + 旋转向量保证单位四元数、无奇异，并让小步预测初始化为恒等。
- 备选：直接预测四元数后归一化。
- 验证：零输出时 ŝ_{t+1} = ŝ_t；梯度有限差分检查。

**D3（U2）状态损失归一化与"名义误差"。**
- 决定：定义每维名义尺度 σ_s：根位置 0.05 m，根旋转向量 0.1 rad，根线速度 0.5 m/s，根角速度 1.0 rad/s，关节位置 0.1 rad，关节速度 2.0 rad/s。状态损失 = 各维 ((ŝ − s)/σ_s)² 的**求和**（镜像论文的 ‖·‖²，保持与 BCE 项的相对权重；BCE 权重另设配置 `term_loss_weight`，默认 1.0）；根部位置/朝向误差在航向系比较。报告的"k 步归一化预测误差"= 各维 ((ŝ − s)/σ_s)² 的均值在 k 步上再取均值，且按论文"measured online on policy rollouts"，在每次采集后的**最新 transition** 上评估，不在训练批上评估。
- 理由：原始单位下关节速度会主导损失。
- 备选：用 replay buffer 的一步差分运行标准差做尺度（配置 `state_scale="running_std"`）。
- 验证：论文 Fig. 3c 的 0.007–0.011 不可直接对比，只比较趋势（随数据量稳定）。文档中明确写出这一点。

**D4（U3）头的选择。**
- 决定：观测头 **解析**（由 ŝ、a_{t−1}、ξ 的关节偏移算出 93 维无噪观测）；奖励头 **解析**（torch 可微 FK 由 qpos 得到 14 个跟踪刚体的位姿，由 qvel 沿运动树传播得到刚体线/角速度，即位置 FK + 速度 FK，二者对 MuJoCo 的 `xpos/xquat/cvel` 都可精确对拍）；终止头 **学习**（MLP (256, 256) → sigmoid，BCE 训练，输入 ŝ_{t+1} 的根高度/投影重力与目标帧的参考根高度/参考投影重力）。
- 理由：解析奖励不可被策略通过"奖励头外推"利用，且 FK 可对 MuJoCo 做机器精度测试；终止是硬阈值，学习头给出平滑概率，符合 Eq. 1 含 BCE 项。参考动作侧的 14 个刚体位姿/速度同样由参考 (qpos, qvel) 经同一 FK 在线算出，仿真奖励与想象奖励共用一条代码路径。
- 备选：学习奖励头 MLP (512, 512)（配置 `reward_head="learned"`）；速度 FK 备选为相邻状态有限差分（配置 `body_vel="finite_diff"`）。终止头备选：解析 sigmoid((|e_z| − 0.25)/0.02) 与倾斜项的最大值。
- 验证：`test_fk_vs_mujoco`（位置、四元数 atol 1e-5；线/角速度 rtol 1e-5 + atol 1e-4；测试内关闭 TF32）；`test_reward_matches_mjlab`（先 `sim.forward()`，再直接调用 mjlab 的奖励函数与我们的实现逐项比较，容差 1e-5，见 Phase 2.8）。

**D5（U5）目标 g_t 的每帧内容与起点。**
- 决定：每帧 68 维 = [参考关节位置 − q⁰ 29, 参考关节速度 29, 参考根部投影重力 3, 参考根部角速度（本体系）3, 参考根线速度（参考航向系）3, 参考根高度 1]。K = 10 帧为当前目标帧 t 及其后 9 帧（t..t+9）。超过 clip 末尾时重复最后一帧。
- 理由：论文观测中没有锚点相对位姿（SPEC §2），策略没有全局位置信息，所以目标必须自带足够的局部动力学信息；不含世界系 xy/yaw，保持平移/旋转不变。"当前目标帧"沿用 mjlab 约定（命令给出下一控制步要达到的帧，奖励对同一帧计算，`mjlab/managers/command_manager.py:111-134`）。
- 备选：加入锚点相对位姿观测（mjlab 默认，需状态估计），配置 `anchor_obs=True`。
- 验证：固定一条 clip 的目标序列可视化对拍。

**D6（U6）动作缩放与裁剪。**
- 决定：α 沿用 mjlab G1 的 `G1_ACTION_SCALE`（0.25 · effort / kp，按关节组 0.075–0.548，`mjlab/asset_zoo/robots/unitree_g1/g1_constants.py:280-288`），PD 增益沿用 mjlab（kp = armature·ω², ω = 2π·10 Hz，阻尼比 2）。动作不裁剪。
- 理由：论文声明遵循 BeyondMimic；mjlab 的跟踪任务即 BeyondMimic 公式。

**D7（U7）奖励参考系与刚体集合。**
- 决定：沿用 mjlab 的精确定义：参考刚体位姿被"重锚"到机器人锚点，但只对齐 **xy 平移与 yaw**（z、roll、pitch 保留参考值），见 `mjlab/tasks/tracking/mdp/commands.py:379-407`；刚体位置/朝向项对重锚后的参考比较，根部位置/朝向项在世界系对锚点 **torso_link** 比较，刚体线/角速度在世界系比较；T = mjlab 的 14 个刚体（`mjlab/tasks/tracking/config/g1/env_cfgs.py:41-57`）。去掉自碰撞项，关节限位权重 −2。解析奖励头必须逐字复制这个重锚定义。
- 注意：奖励的"根"是锚点 torso_link（BeyondMimic / mjlab 约定），而论文终止条件明确写 "robot root (pelvis)"，两处用不同刚体。终止刚体进配置 `termination_body`，默认 `pelvis`（按论文），备选 `torso_link`（mjlab 默认）。
- 理由：同 D6。

**D8（U8）关节标定偏移与摩擦语义；ξ 的采样时机与存储。**
- 决定：关节标定偏移用 mjlab 的 `dr.encoder_bias`（`mjlab/envs/mdp/dr/joint.py:209-241`）。经核实它同时作用于两端：观测侧 `joint_pos_biased = joint_pos + encoder_bias`（`mjlab/entity/data.py:390-392`，跟踪任务的 `joint_pos_rel` 以 `biased=True` 使用它），执行侧 `JointPositionAction.apply_actions` 把目标减去 `encoder_bias`（`mjlab/envs/mdp/actions/actions.py:221-223`），即"编码器零点偏差"的完整语义。**不要用** `dr.joint_default_pos`：它只写 MuJoCo 的 `qpos0`（`dr/joint.py:178-202`），而动作偏置与 `joint_pos_rel` 读的都是实体初始化时复制的 `default_joint_pos`（`actions.py:218-219`，`mjlab/envs/mdp/observations.py:51-61`），改 `qpos0` 对观测与设定点都无效。
- 采样时机：摩擦与 encoder_bias 用 `mode="reset"` 按回合重采样（两者都不触发常量重算，`dr/geom.py` 的 `geom_friction` 为 `RecomputeLevel.none`，`encoder_bias` 是纯张量写）。躯干 COM 偏移保持 `mode="startup"`：`body_com_offset` 标记 `RecomputeLevel.set_const`（`mjlab/envs/mdp/dr/body.py:328`），任一 reset 事件触发后 `EventManager` 会对**全部**世界做一次 `sim.recompute_constants`（`mjlab/managers/event_manager.py:332-337`，`mjlab/sim/sim.py:479-490`），4096 环境下几乎每步都有重置，等于每步一次全局 set_const。摩擦只作用于足部碰撞几何体，所有足部几何体共享同一个值（mjlab 默认）。replay buffer 按**步**存 ξ（33 维），避免回合边界簿记。
- 理由：ξ 中两项按回合变化、一项按环境固定，世界模型只需要每步拿到正确的 ξ，不关心采样时机。
- 备选：COM 用自定义事件原地写 `body_ipos` 且不触发重算（接受 `invweight0` 微小过期），配置 `com_dr_mode="reset_no_recompute"`；若 reset 模式整体过慢，退回全部 `startup`。
- 验证：Phase 0.3 的吞吐基准必须使用真实 DR 配置；观测头测试覆盖带偏移的情形；`sim_context.py` 读回的 ξ 与事件采样值一致。

**D9（U9）上一动作与动作噪声的处理。**
- 决定：观测中的 a_{t−1} 与动作平滑项用的都是策略输出的**干净**动作 a_t = π_θ(o_t, g_t)；动作噪声 ε_t 只在执行时加到设定点上，只进入模拟器与 replay buffer 的 `a_exec` 列（世界模型拟合用 `a_exec`）。实现：自定义动作项 `NoisyJointPositionAction`（继承 mjlab `JointPositionAction`），`process_actions` 记录干净动作（供 `last_action` 观测项与 `action_rate_l2` 使用，`mjlab/managers/action_manager.py:166-169`），`apply_actions` 在干净动作上加 ε_t 后再计算设定点，并把执行动作暴露给采集器。a_{t−1} 作为随 rollout 携带的"扩展状态"，不由 f_ω 预测；观测头签名 g(ŝ_t, a_{t−1}, ξ)。回合起点 a_{−1} = 0（mjlab 重置时清零 `prev_action`，`action_manager.py:140-146`），replay buffer 在 `ep_start` 处也取 0。
- 理由：论文 §3.1 的记号是 a_t = π_θ(o_t, g_t)、模拟器执行 a_t + ε_t、噪声"serves only the model fit"；Table A.1 的 a_{t−1} 是策略量。这样采集、想象、部署三处的策略输入分布一致，想象起点从 buffer 重建的 9 帧历史也不会混入噪声。
- 备选：把带噪执行动作放入观测（配置 `obs_prev_action="executed"`）。

**D10（U10）critic 输入与梯度流清单。**
- critic 输入：D1 的航向规范化状态（71）+ g_t（680）+ ξ（33）+ **根部相对误差 e_root（4）** = 788 维。e_root = 机器人根部相对参考根部的位置误差（在参考航向系，3）与 yaw 误差（1），由 ŝ 与参考帧的世界系根位姿算出。原因：奖励的根部位置/朝向项（每步最大奖励的 1/5）依赖世界系漂移，而规范化状态与 g_t 都不含 xy/yaw；不加这 4 维，V_ψ(ŝ_H) 对漂移的梯度恒为零，H 之外的根部回报信号全部丢失。策略按论文设计看不到这项（Table A.1），critic 作为训练期特权量可以看到。
- 梯度流清单：
  1. Eq. 2 对 θ 的梯度经 ŝ_t、a_t、ō̂_t、ĉ_t 全程 BPTT；ĉ_t 对 d̂（经终止头到 ŝ）**保留梯度**（论文："keeps the survival weight differentiable"）。
  2. 终值 V_ψ(ŝ_H, …) 对 ŝ_H **保留梯度**（SHAC 核心），对 ψ 不传梯度。
  3. 拟合 critic 时输入与 GAE 目标全部 detach；目标每个 rollout 算一次后固定。
  4. Eq. 1 的 k 步自回归拟合在 8 步内全程 BPTT，不 detach 中间预测状态。
  5. 策略更新内 ω 冻结（`requires_grad_(False)` 或 stop-gradient），但保留对输入的反传。
- 理由：论文"目标固定""ω 冻结"的表述与 SHAC 一致。

**D11（U11）激活、归一化、初始化。**
- 决定：全部 MLP 用 SiLU，无归一化层，默认初始化；动力学最后一层零初始化（残差恒等起步）；解码器最后一层增益 0.01。
- 理由：论文未指明，取稳妥默认。
- 备选：ELU（mjlab PPO 默认）、LayerNorm。

**D12（U12）UTD 与迭代定义。** 论文无法唯一判定，拆成"确定部分"与"待校准部分"。
- 确定部分（调度比例）："Update-to-data ratio 0.4" 按向量化 RL 的通行定义 = 每一次全部 4096 环境的仿真步对应 0.4 次更新，即**平均每 2.5 个仿真步做 1 次更新**（1 次更新 = 10 个世界模型梯度步 + 1 次策略更新 + critic 拟合，"Policy updates per world-model update = 1"）。实现为小数累加器（acc += 0.4 每仿真步；acc ≥ 1 时更新）。Algorithm A.1 的 "Both ω and θ update every iteration" 在"迭代 = 更新、其采集阶段平均含 2.5 步"的读法下成立。
- 待校准部分（总长度）：Table A.5 的 "320,000 iterations" 可能指 (A) 仿真步数（⇒ 128k 次更新）或 (B) 更新次数（⇒ 800k 仿真步）；也不能排除 (C) 作者的"迭代"是 1 步 + 1 次更新而 0.4 另有所指。三者的梯度更新总数相差 2.5×。时间自洽性论证不能区分它们：Fig. 5a 的仿真吞吐是对数坐标图读（±2×），"单卡约 10 h"也只是 Fig. 3a / A.1 横轴的图读。因此：`updates_per_env_step=0.4` 与 `total_env_steps` 都进配置，默认按 (A)；学习率余弦调度按**比例**定义（总长度的 250/320 处衰减到 0.03×），与总长度解耦。
- 校准方法：Phase 4.7 的 e2 以论文的两个正文锚点反推——LAFAN1 单卡 1.2 h 到评测回报 0.82；BONES-SEED 单卡 3 h 超过 SONIC。把我们在 5090 上的到达时间按第 9 节的算力比折算，若更新总数明显不足（曲线仍在上升即被调度衰减压住）则切到 (B)。
- 验证：记录每迭代的仿真 / 世界模型 / 策略耗时分布。

**D13（U13）多 GPU 归属。**
- 决定：每卡独立 4096 环境与独立 replay buffer，世界模型/策略/critic 用 DDP 平均梯度，学习率 × √world_size，种子 + rank。
- 理由：论文 2 卡学习率 × √2 暗示有效批翻倍，即每卡各自采集。

**D14（U14）基于回报的重置采样细节。**
- 决定：每条 clip 的时间线按 1 s 切 bin，全库 bin 全局编号。G_max 按论文取**单一常量** = 可达到的最大折扣回报 = 0.1 / (1 − γ) = 10。每回合结束时对回合内每一步反向计算折扣回报-to-go R_t，按步所在 bin 做 EMA 更新 G_b（α = 0.05，初值 G_max）。**截断处理**：若回合因 10 s 超时结束（D15），回报-to-go 用 critic 自举 R_T = V_ψ(s_T, g_T, ξ, e_root)（critic 输入在仿真状态上同样可得），避免临近超时的步被系统性低估为"难"；若回合因 clip 结束而止，不自举（参考真的结束了），并接受末尾 bin 的 G_b 天然偏低。混合：50% 均匀、40% softmax(−25 G_b/G_max)、10% ∝ 该 bin 的历史 softmax 权重（EMA α = 0.01）。bin 内起帧均匀。重置时加 mjlab 默认姿态噪声（关节 ±0.1 rad，根位姿 xy ±0.05 m、z ±0.01 m、roll/pitch ±0.1、yaw ±0.2，`mjlab/tasks/tracking/tracking_env_cfg.py:144-153`）。
- 备选：不加重置噪声（配置 `reset_noise=False`）；按 bin 剩余步数归一化（配置 `gmax_mode="per_bin"`）；不自举、只用剩余步数 ≥ 100 的步更新 G_b（配置 `truncation="drop"`）。
- 验证：`test_reset_sampler`：难 bin（低 G_b）被采到的频率显著高于均匀；超时前 1 s 内的 bin 不应因截断被标为难。

**D15（U15, U19, U20）回合长度与 clip 末尾。**
- 决定：仿真回合在 clip 末尾以 **timeout**（非终止）结束，另设最大长度 10 s timeout；不做 mjlab 的"到末尾后 teleport 到新 clip"（`mjlab/tasks/tracking/mdp/commands.py:421-423`）。replay buffer 记录 episode_start 标志，k 步窗口与想象起点不跨越回合边界。想象中若超过 clip 末尾，目标保持最后一帧，继续滚动与计奖励。评测回合 = 整条 clip（无 10 s 上限）。
- 理由：teleport 会在 buffer 中制造不可建模的跳变；想象中保持末帧最简单且动力学仍有效。
- 备选：想象超末尾处 ĉ 置零（配置 `imag_clip_end="mask"`）。

**D16（U16）过滤规则补全。**
- SONIC 的关键词表已找到：`NVlabs/GR00T-WholeBodyControl` 仓库 `gear_sonic/data_process/filter_and_copy_bones_data.py`（commit 45c03fa，2026-10-13），默认 39 个子串，对文件名小写做包含匹配：bed, bike, chair, climb, com_up_50cm, sitting, step_on, seat, table, _sit_, sit_, ladder, crutch, _bed_, _ride_, scooter, stepdown, acrobatics_, box_HSPU, cartwheel, 50cm_box_, on_box, fall_from, handstand_ff_, on_1m, form_box, off_1m, 230m, jump_over_obstacle_, lift_crate_come_up_, jump_to_shoulder_roll, kozak_dance, stair, handstand, box_jump, monkey_jump, safety_roll, box_dips, walking_on_edge, push_obstacle。
- 决定：论文说其表"改编自 SONIC 的表并补充遗漏项"，且只针对"地形或物体接触"。而 SONIC 的表还含纯技巧类词（acrobatics_, cartwheel, handstand, handstand_ff_, safety_roll, kozak_dance, jump_to_shoulder_roll, monkey_jump），与论文 benchmark 保留 44 个翻转、22 个侧滚相矛盾。因此我们的表 = SONIC 表去掉这 8 个技巧词 + 论文例子中的词。**匹配方式**与 SONIC 脚本一致：文件名小写后做**子串**匹配；新增词一律写成下划线定界的形式以避免误杀（`_box_`、`box_`、`_sit_`、`sit_`、`_stair`、`ladder`、`chair`、`climb`、`jump_off_50cm`、`off_50cm`、`fence`），不加裸词 `sit`/`box`/`ball`/`table`/`wall`（会命中 position、boxing、ballet、stable）。校准门槛按 Table A.7 的三个数字分别设：名字过滤后 129,667（±0.5%）、可行性过滤后 129,514（±0.5%）；超出时输出每个关键词单独命中的 clip 数与样例文件名，逐词排查并记录。
- 可行性过滤的门限：直立 = 骨盆投影重力 z 分量 > 0.9；站立高度 = 骨盆 z ∈ [0.60, 0.85] m；近乎静止 = 骨盆速度范数 < 0.1 m/s 且关节速度 RMS < 0.5 rad/s。所有阈值进配置并记录过滤后数量与论文 Table A.7 对比（目标 129,514 ± 1%）。

**D17（U17）LAFAN1 子集与长度子集。**
- 决定：LAFAN1 用 HuggingFace `lvhaidong/LAFAN1_Retargeting_Dataset` 的 `g1/` 目录全部 clip（30 fps CSV）。已核实该目录恰好 **40 个 CSV**（dance1/2、fallAndGetUp1–3、fight1、fightAndSports1、jumps1、run1/2、sprint1、walk1–4 的若干 subject），与论文的 40 条一致；转换后核对总时长是否为 2.45 h。Fig. 5b 的 walk/run/sprint 子集 = 名字以 walk/run/sprint 开头的 18 条。数据长度子集（1, 7, 40 h）按 clip 随机抽取，固定种子。

**D18（U18）速度编码器的观测历史特征。** M6 可选，届时决定：默认把 930 维历史观测过一个 MLP (256) 得到特征。

**D19 精度。** matmul 开 TF32（mjlab 默认 `mjlab/utils/torch.py`），状态积分与奖励在 fp32；不用 bf16。

**D20 观测历史的布局与初始化。**
- 布局：沿用 mjlab 的 **term-major** 展平顺序 `[A_t0..A_t9, B_t0..B_t9, ...]`（`mjlab/managers/observation_manager.py:63-65, 380-383`），而不是逐帧拼接的 10×93。想象侧直接复用 `mjlab.utils.buffers.CircularBuffer` 并以同一顺序展平，保证策略权重在仿真与想象间可互换。
- 分组：本体观测五项放观测组 `policy`，组级 `history_length=10`；目标项（680 维）放**独立观测组** `goal`，不带历史。原因：组级 `history_length` 会覆盖组内所有项（`observation_manager.py:461-463`），若目标项同组会被堆成 6800 维。
- 初始化：回合开始时用首帧观测回填 10 帧历史（`CircularBuffer` 的 backfill 语义）；想象起点的历史从 replay buffer 同一环境槽位的前 9 步取，跨回合边界处用起点观测回填。

---

## 5. Repo 结构（实施骨架）

```
SuperMimic/
├── README.md
├── pyproject.toml                 # uv 项目；mjlab 固定到 git commit 8bb2bc9；自声明 torch/warp 索引；入口脚本
├── uv.lock
├── Makefile                       # make test / test-fast / lint / smoke
├── .gitignore                     # logs/, data/, *.npz, wandb/
├── docs/
│   ├── paper/                     # 论文 PDF 与全文
│   ├── PAPER_SPEC.md
│   ├── REPRODUCTION_PLAN.md       # 本文件
│   ├── REVIEW_LOG.md
│   └── EXPERIMENTS.md             # 实验日志（实施期填写）
├── src/supermimic/
│   ├── __init__.py
│   ├── config.py                  # 全部 dataclass 配置：DataCfg, EnvCfg, WorldModelCfg,
│   │                              #   PolicyCfg, CriticCfg, ImaginationCfg, TrainCfg, EvalCfg
│   ├── presets/
│   │   ├── lafan1.py              # LAFAN1 40 clip 预设
│   │   ├── bones_seed.py          # 全量预设（含多卡学习率缩放）
│   │   └── smoke.py               # 单 clip、小网络、几百迭代
│   ├── envs/
│   │   ├── __init__.py            # make_env(cfg, device) -> ManagerBasedRlEnv；不走 mjlab registry
│   │   │                          #   （registry 强制要求 rsl_rl 的 rl_cfg，mjlab/tasks/registry.py:22-28）
│   │   ├── tracking_env_cfg.py    # mjlab 环境配置：观测(SPEC §2，policy 组带历史 + goal 组)、奖励(§5)、
│   │   │                          #   终止(§6)、DR(§7)、无 push
│   │   ├── actions.py             # NoisyJointPositionAction：干净动作入观测，执行时加 ε_t（D9）
│   │   ├── sim_context.py         # 从 sim 读取 ξ(33) 张量；DR 事件写入后的采集
│   │   ├── state_io.py            # StateIO: read_state()->[N,71]; write_state(ids, s); forward()
│   │   └── mdp/
│   │       ├── motion_library.py  # MotionLibrary: 多 clip 打包、clip 边界、frame 索引、gather
│   │       ├── commands.py        # MultiClipMotionCommand: 每环境(clip, t)、未来 K 帧目标、
│   │       │                      #   回合在 clip 末尾超时；reset_to(clip, frame)
│   │       ├── reset_sampler.py   # ReturnBasedResetSampler（D14）
│   │       ├── observations.py    # projected_gravity、goal_future_frames 观测项
│   │       ├── rewards.py         # tracking_reward(): 仿真与想象共用的 torch 实现；
│   │       │                      #   仿真侧包装为 mjlab RewardTerm
│   │       └── terminations.py    # root_height_dev / root_tilt（骨盆）
│   ├── kinematics/
│   │   ├── g1_fk.py               # 从 MJCF 解析运动树；torch 可微位置 FK + 速度 FK；跟踪刚体子集；
│   │   │                          #   四元数逐元素运算（不用 3×3 bmm，避免 TF32 误差）
│   │   └── rotations.py           # 薄封装：复用 mjlab.utils.lab_api.math 的 quat_mul/quat_inv/
│   │                              #   quat_apply/yaw_quat（奖励与命令项用的同一套约定），只新增
│   │                              #   rotvec exp/log、6D、航向系提取
│   ├── world_model/
│   │   ├── state_spec.py          # 71 维布局、名义尺度 σ_s、canonicalize/uncanonicalize、integrate(Δ)
│   │   ├── dynamics.py            # DynamicsMLP (2048×3)，残差输出（D2）
│   │   ├── heads.py               # ObsHead(解析)、RewardHead(解析 FK / 学习)、TermHead(学习)
│   │   ├── world_model.py         # WorldModel.step(ŝ, a, a_prev, ξ, g) -> (ŝ', ō̂, r̂, d̂)
│   │   └── wm_trainer.py          # k 步窗口损失 Eq.1；10 步 × 2048 窗口；梯度裁剪；指标
│   ├── policy/
│   │   ├── mlp.py                 # 通用 MLP（SiLU）
│   │   ├── encoders.py            # MotionEncoder(680→32 tanh)、VelocityEncoder(M6)
│   │   ├── decoder.py             # Decoder(z + 930 → 29)
│   │   ├── policy.py              # Policy: interfaces dict → act(obs_hist, goal, iface)
│   │   ├── obs_history.py         # 想象侧历史：复用 mjlab CircularBuffer，term-major 展平（D20）
│   │   └── critic.py              # Critic(ŝ, g, ξ) MLP (256,128)，每接口一个
│   ├── imagination/
│   │   ├── rollout.py             # imagine(start_batch, H): 噪声重加、存活权重、目标帧推进
│   │   ├── objective.py           # policy_objective (Eq.2)、gae_targets (Eq.3)
│   │   └── policy_trainer.py      # BPTT 更新 + critic 5×4 拟合 + 裁剪（范数 1.0 → 逐元素 0.1）
│   ├── data/
│   │   ├── replay_buffer.py       # GPU 环形 [env, T]：s, a(干净), a_exec, ξ, r, d, clip, frame,
│   │   │                          #   ep_start, terminal 标志；回合结束时在重置前多写一行 (s_{T+1}, d=1,
│   │   │                          #   terminal=True) 供 Eq.1 的终止正样本与末步状态目标使用；
│   │   │                          #   采样器不以 terminal 行为窗口起点或想象起点
│   │   │                          #   sample_windows(k) / sample_starts(n)，不跨回合边界
│   │   └── collector.py           # 仿真 rollout：动作噪声、观测噪声、写 buffer、触发重置采样；
│   │                              #   用 RecorderManager 的 record_pre_reset 钩子在重置前读终态 s_{t+1}
│   │                              #   （比 auto_reset=False 省一次 forward），奖励在 step 后自行重算
│   ├── interfaces/
│   │   ├── base.py                # GoalInterface 协议：encoder、goal_for(ŝ_batch, t)、reward、term、critic
│   │   ├── motion_tracking.py
│   │   └── velocity.py            # M6
│   ├── training/
│   │   ├── loop.py                # Algorithm A.1；UTD 累加器（D12）
│   │   ├── schedule.py            # 余弦学习率（250k → 0.03×）
│   │   ├── distributed.py         # DDP 初始化、√world_size 学习率、rank 种子
│   │   ├── checkpoint.py          # 保存/恢复：ω, θ, ψ, 优化器, 采样器状态, 迭代数
│   │   └── logger.py              # TensorBoard/W&B：各损失、k 步归一化误差、训练回报、耗时分解
│   ├── eval/
│   │   ├── benchmark.py           # 每条动作从首帧播放一次；摩擦 1.0；无 DR/噪声；失败判定；归一化回报
│   │   ├── normalized_return.py
│   │   ├── push_test.py           # T-pose 推力 0.5–4 m/s，稳定时间
│   │   └── step_response.py       # M6
│   ├── export/
│   │   └── onnx_export.py         # M6：编码器+解码器+历史缓冲 → ONNX
│   └── utils/
│       ├── timing.py              # 分段计时
│       └── compile.py             # torch.compile(mode="reduce-overhead") / CUDA graph 封装（想象步、FK）
│                                  # 随机种子直接用 mjlab.utils.random.seed_rng
├── scripts/
│   ├── data/
│   │   ├── download_lafan1.py
│   │   ├── download_bones_seed.py # 需用户先在 HF 接受许可
│   │   ├── csv_to_npz_batch.py    # 本地批量：30/120 fps → 50 Hz，FK，写 npz；复用 mjlab 的重采样函数
│   │   ├── filter_bones_seed.py   # Appendix E 两道过滤 → manifest.csv（含过滤原因）
│   │   └── build_library.py       # 打包为 library.npz（+ clip_index.json）
│   ├── train.py                   # tyro CLI：--preset lafan1 --num-gpus 1
│   ├── eval_benchmark.py
│   ├── eval_push.py
│   ├── bench_wm_speed.py          # Fig. 5a 对应
│   ├── bench_sim_speed.py         # 仿真吞吐（Phase 0）
│   └── plots/
│       ├── fig3_return_vs_gpuh.py
│       ├── fig4_horizon_sweep.py
│       └── fig5_speed.py
├── tests/
│   ├── conftest.py                # 小环境 fixture（64 envs）
│   ├── test_rotations.py
│   ├── test_fk_vs_mujoco.py
│   ├── test_state_io_roundtrip.py
│   ├── test_state_spec.py         # canonicalize ↔ uncanonicalize；integrate 零增量
│   ├── test_obs_head_matches_env.py
│   ├── test_reward_matches_mjlab.py
│   ├── test_terminations.py
│   ├── test_motion_library.py     # 边界、gather、末帧保持
│   ├── test_replay_buffer.py      # 窗口不跨回合；环形覆盖
│   ├── test_reset_sampler.py
│   ├── test_world_model_overfit.py# 小数据过拟合到低误差
│   ├── test_bptt_gradients.py     # 微型模型有限差分
│   ├── test_gae_targets.py        # 对照手算
│   ├── test_filters.py
│   └── test_train_smoke.py        # smoke 预设 20 迭代
├── experiments/                   # 每个实验一个目录：config.json、notes.md（输出 gitignore）
│   ├── e0_smoke/
│   ├── e1_lafan1_single_clip/
│   ├── e2_lafan1_40/
│   ├── e3_horizon_sweep/
│   ├── e4_wm_speed/
│   └── e5_bones_seed/
├── data/                          # gitignore：raw/、npz/、library/
└── logs/                          # gitignore
```

关键接口签名（实施时以此为准，变更须更新本节）：

```python
# envs/state_io.py
class StateIO:
    def read(self) -> torch.Tensor                       # [N, 71] fp32，qpos|qvel
    def write(self, env_ids: Tensor, s: Tensor) -> None  # 写 qpos/qvel 后 sim.forward()
# envs/sim_context.py
def read_sim_context(env) -> torch.Tensor                # [N, 33]
# world_model/world_model.py
class WorldModel(nn.Module):
    def step(self, s, a, a_prev, xi, goal_frame) -> StepOut  # s_next, obs_clean, reward, term_prob
    def loss(self, window: WindowBatch) -> tuple[Tensor, dict]  # Eq.1 + 分项指标
# policy/critic.py
class Critic(nn.Module):
    def forward(self, s_canon, goal, xi, e_root) -> Tensor   # e_root: 参考航向系根位置误差(3) + yaw 误差(1)
# imagination/rollout.py
def imagine(wm, policy, starts: StartBatch, H: int, noise: ObsNoise) -> ImagRollout
#   ImagRollout: s[H+1], r[H], d[H+1], c[H+1], g[H+1], a[H]
# imagination/objective.py
def policy_objective(ro: ImagRollout, critic, gamma) -> Tensor      # Eq.2（标量，取负做 loss）
def gae_targets(ro: ImagRollout, critic, gamma, lam) -> Tensor      # Eq.3 [H, B]
# data/replay_buffer.py
class ReplayBuffer:
    def add_step(self, s, a, a_exec, xi, r, d, clip, frame, ep_start) -> None
    def add_terminal(self, env_ids, s_next) -> None            # 重置前写入终态行
    def sample_windows(self, n: int, k: int) -> WindowBatch
    def sample_starts(self, n: int, hist: int) -> StartBatch
```

---

## 6. Step-by-step 实施计划

每步列出：任务、产出、验证、估时。验证不过不进入下一步。

### Phase 0：环境与基线（1–2 天）

| # | 任务 | 产出 | 验证 |
|---|---|---|---|
| 0.1 | 建 uv 项目，按 §2.3 声明索引并安装 mjlab（git commit 8bb2bc9，cu130） | `pyproject.toml`, `uv.lock` | `uv run list-envs` 列出 G1 跟踪任务；`torch.version.cuda` 为 13.0 |
| 0.2 | 跑 mjlab 自带 G1 跟踪 PPO 冒烟（一条 LAFAN1 clip，需先完成 1.1 的单 clip 转换），300 迭代 | 日志 | 回报上升，无 NaN |
| 0.3 | `bench_sim_speed.py`：1k/2k/4k/8k 环境下 MuJoCo Warp 步/s | 表格写入 EXPERIMENTS.md | 用于校准第 9 节预算 |
| 0.4 | 机器 B 驱动升级（B1）；升级后重复 0.1 | – | 四卡 `uv run list-envs` 通过 |
| 0.5 | 初始化 git 仓库，提交骨架（空模块 + 文档） | 首个 commit | – |

### Phase 1：数据管线（2–3 天，可与 Phase 2 并行）

| # | 任务 | 产出 | 验证 |
|---|---|---|---|
| 1.1 | `csv_to_npz_batch.py`：读 CSV（无表头，列 0–2 位置、3–6 四元数 xyzw、后 29 关节，`mjlab/scripts/csv_to_npz.py:62-64,398-428`），lerp/slerp 重采样到 50 Hz，FK 得 30 刚体位姿与速度；写本地 npz，键与 mjlab `MotionLoader` 一致（`joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w, fps`）。**不要**直接调用 mjlab 脚本的逐帧路径：其 `_slerp` 是逐帧 Python 循环、`run_sim` 每帧一次 `forward()` 加 6 次 `.cpu()`，并硬编码 `/tmp/motion.npz` 与 W&B（`csv_to_npz.py:108-115, 265-281, 312-329`）。改为批量：slerp 向量化；FK 用 CPU `mujoco`（`mj_kinematics` + `mj_comPos`）在 128 核多进程下逐 clip 计算，或把一条 clip 的全部帧当作并行世界一次 `forward()` | `data/npz/<clip>.npz` | 单 clip 与 mjlab 原脚本（patch 为本地输出）对拍，逐键 ≤ 1e-5，帧数相等 |
| 1.2 | 下载 LAFAN1 G1 CSV（30 fps，40 个文件）；全部转换；统计总时长 | `data/npz/lafan1/` | 40 条；时长与论文 2.45 h 对照并记录 |
| 1.3 | `MotionLibrary`：打包所有 clip 为连续张量，每帧只存 71 维参考 (qpos, qvel)（BONES-SEED 全量约 47.6M 帧 × 71 × 4 B ≈ 13.5 GB）；目标帧的 68 维与奖励需要的 14 刚体位姿/速度都由该 71 维经 `g1_fk` 在线算出（每迭代 4096 × (1 + 32) 帧，开销可忽略）；clip 起止表；`gather(clip, frame, K)` 含末帧保持；GPU 常驻（LAFAN1 约 125 MB）或 CPU pinned + 按需 gather（BONES-SEED） | `library.npz` | `test_motion_library` |
| 1.4 | 申请 BONES-SEED 访问（用户在 HF 接受许可）；下载 `g1.tar.gz` 与 metadata。**硬门**：确认 G1 CSV 的 (a) 每 clip fps（从 metadata 逐 clip 取，不假定全局 30 或 120）、(b) 29 关节的顺序是否与 mjlab/LAFAN1 的深度优先顺序一致、(c) 四元数是 xyzw 还是 wxyz、(d) 是否有表头。任一不同都要在转换器里做映射并用 FK 可视化验证 | `data/raw/bones_seed/` + 格式说明写入 EXPERIMENTS.md | 抽样 20 条 clip 转换、回放、目视无肢体错位 |
| 1.5 | `filter_bones_seed.py`：名字过滤 + 可行性过滤（D16），输出 manifest 与各过滤原因计数 | `manifest.csv` | 数量对照 Table A.7（142,220 → ≈129,667 → ≈129,514） |

### Phase 2：环境、状态 I/O 与可微 FK（3–4 天）

| # | 任务 | 产出 | 验证 |
|---|---|---|---|
| 2.1 | `rotations.py`：复用 `mjlab.utils.lab_api.math`，新增 rotvec exp/log、6D、航向提取 | 模块 | `test_rotations`（对照 scipy，float64） |
| 2.2 | `g1_fk.py`：从 `g1.xml` 解析树（body 父子、关节轴、位置偏移），torch 批量位置 FK + 速度 FK，四元数逐元素实现 | 模块 | `test_fk_vs_mujoco`：测试内关闭 TF32（`torch.backends.cuda.matmul.allow_tf32=False`）；随机 qpos/qvel 下 30 刚体位置/四元数 atol 1e-5，线/角速度 rtol 1e-5 + atol 1e-4；反传可用 |
| 2.3 | `tracking_env_cfg.py`：观测组 `policy`（93 维五项，组级历史 10，term-major）与 `goal`（680 维，无历史）；动作项 `NoisyJointPositionAction`（D9）；奖励（去自碰撞、关节限位 −2）；终止（骨盆高度/倾斜，无末端终止）；事件（足摩擦 abs U[0.3,1.2] 共享 reset 模式、`encoder_bias` U[−0.01,0.01] reset 模式、躯干 COM add startup 模式；删除 push）；sim 0.005 s × 4；回合 10 s | `make_env()` | 环境能 reset/step；观测维度 930 + 680；`last_action` 观测等于干净动作而执行设定点含噪声 |
| 2.4 | `MultiClipMotionCommand`：每环境 (clip, t)；步进；末尾超时（以"本步被奖励的帧 == 末帧"判定，因 `step()` 先算终止再推进命令，`mjlab/envs/manager_based_rl_env.py:452, 486-488`）；重置路径保留 mjlab 的 `+1` 约定（`commands.py:416-420`）；`reset_to(clip, frame)`；目标帧 gather | 模块 | `test_motion_library` 扩展 |
| 2.5 | `ReturnBasedResetSampler`（D14） | 模块 | `test_reset_sampler` |
| 2.6 | `StateIO`：读 `sim.data.qpos/qvel` 拼 71 维；写回 qpos/qvel 与 `qacc_warmstart` 后 `sim.forward()`（`mjlab/sim/sim.py:492`）；注意 CUDA graph 下只做原地写 | 模块 | `test_state_io_roundtrip`：写入任意合法状态后读回逐元素相等；写入 s_t 并执行记录的 a_t 后，所得 s_{t+1} 与 buffer 记录值的归一化误差 p99 ≤ 1e-3（Newton 10 次迭代在富接触帧未必收敛，按分位数而非最大值判定） |
| 2.7 | `sim_context.py`：DR 事件执行后读取每环境 ξ（摩擦从 `model.geom_friction` 的 per-world 字段，关节偏移从 `entity.data.encoder_bias`，COM 从 `body_ipos` 与名义值之差） | 模块 | 与事件采样值一致 |
| 2.8 | `rewards.py` 共用实现 + `test_reward_matches_mjlab`：测试中先 `sim.forward()`，再直接调用 mjlab 的 `motion_*_error_exp` 函数（它们只读命令项属性，`mjlab/tasks/tracking/mdp/rewards.py:26-113`）与我们的实现逐项比较，绕开 `env.step` 路径里"运动学晚一个子步"的差异（`manager_based_rl_env.py:395-410`） | 模块 | 6 个跟踪项 + 2 个正则项逐步 ≤ 1e-5 |
| 2.9 | 观测头解析实现 + `test_obs_head_matches_env`：无噪声时与环境观测管理器输出逐元素相等（含 encoder_bias）。IMU site `imu_in_pelvis` 只有位置偏移没有旋转（`g1.xml:69`），基座角速度即骨盆系 qvel[3:6]，无需变换；投影重力按骨盆系 | 模块 | ≤ 1e-5 |

**Phase 2 注意事项（来自 mjlab 调研）**：
- 奖励/终止在 mjlab 的 `env.step` 中看到的运动学比最终状态晚一个子步（`mjlab/envs/manager_based_rl_env.py:395-410`）。采集器不存 mjlab 算出的奖励，而是在 step 返回后（此时已 forward）用我们的 `tracking_reward()` 对终态重算 r，与想象中的奖励定义完全一致；mjlab 的奖励管理器只保留用于日志对照。
- 自动重置会返回重置后的观测（`manager_based_rl_env.py:389-393`）。采集器用 `RecorderManager` 的 `record_pre_reset` 钩子：此时 `_reset_idx` 尚未执行（`:470-471`），`sim.data.qpos/qvel` 就是终态 s_{t+1}。不用 `auto_reset=False`，它会让每次 `reset(env_ids)` 多跑一次 `forward()` 图（`:371-373`），而 4096 环境几乎每步都有重置。
- 重置会把关节裁到软限位（`commands.py:312-313`），写任意记录状态要走 `StateIO.write` 而非命令项的重置。

### Phase 3：世界模型（3–5 天）

| # | 任务 | 产出 | 验证 |
|---|---|---|---|
| 3.1 | `state_spec.py`：布局、σ_s、canonicalize/uncanonicalize、integrate | 模块 | `test_state_spec` |
| 3.2 | `dynamics.py` + `heads.py` + `world_model.py` | 模块 | 形状测试；零增量恒等 |
| 3.3 | `replay_buffer.py` + `collector.py`：用随机/PPO 策略采集 LAFAN1 数据 | 模块 | `test_replay_buffer` |
| 3.4 | `wm_trainer.py`：Eq. 1，k = 8，10 × 2048 窗口，Adam 2e-4，范数裁剪 1.0；训练指标：各维归一化误差、BCE；在线指标：每次采集后在最新 transition 上评估 1/8/32 步归一化误差（D3） | 模块 | `test_world_model_overfit`（1k 窗口过拟合到归一化误差 < 1e-3） |
| 3.5 | 离线实验 e-wm：在固定 buffer 上训练 10k 步；评估 8 步误差、256 步自由滚动是否发散；对比 `canonicalize` 开/关 | EXPERIMENTS.md | 8 步归一化误差 O(1e-2)，256 步不发散 |
| 3.6 | **启动开销控制（必做）**：想象单步 = 动力学 MLP + integrate + canonicalize + 观测头 + 解析 FK 奖励，是数百个小 kernel；×32 步 × 前反向每次更新约 2 万次 launch，而 mjlab 仿真侧已是 CUDA graph 回放（`mjlab/sim/sim.py:354-386`）。对想象单步与 FK 做 `torch.compile(mode="reduce-overhead")`（或手工 CUDA graph），FK 按刚体层级批量化。先做再测速 | `utils/compile.py` | 编译前后对比记录 |
| 3.7 | `bench_wm_speed.py`：512–64k 批下世界模型步/s vs 仿真（Fig. 5a），含 DR 的真实仿真配置 | 表格 | 4k 批 ≥ 5× 仿真（5090 上张量核较弱，目标低于论文的 11×） |

### Phase 4：策略 BPTT 训练（4–6 天）

| # | 任务 | 产出 | 验证 |
|---|---|---|---|
| 4.1 | `policy/`：MLP、编码器、解码器、历史缓冲、critic | 模块 | 形状测试 |
| 4.2 | `imagination/rollout.py`：起点取 (s_0, ξ, clip, frame, 9 步历史, a_{−1})；每步观测头 + 噪声 → 策略 → f_ω；存活权重；目标帧推进（末帧保持） | 模块 | 与世界模型单步一致 |
| 4.3 | `objective.py`：Eq. 2、Eq. 3 | 模块 | `test_gae_targets` 手算对照；`test_bptt_gradients` 在 float64、微型配置（H=3，B=2，宽 16）下做有限差分 |
| 4.4 | `policy_trainer.py`：BPTT 更新（Adam β=(0.9,0.95)，学习率同 α_ω，范数 1.0 → 逐元素 0.1）；critic：Adam（默认 β），学习率同 α_ω，范数裁剪 1.0，每次更新 5 个 epoch，每个 epoch 把 H×B = 32×4096 个 (ŝ_t, g_t, ξ, e_root, R̂_t) 对打乱分成 4 个 minibatch | 模块 | – |
| 4.5 | `training/loop.py`：Algorithm A.1，UTD 累加器，调度，日志，检查点 | `scripts/train.py` | `test_train_smoke` |
| 4.6 | e1：单 clip（walk）训练 2k 迭代 | 曲线 | 训练回报明显上升；仿真回报与想象回报差 < 10%（模型利用监控） |
| 4.7 | e2：LAFAN1 40 clip，默认超参，单卡 5090 | 曲线、检查点 | 训练回报 ≥ 0.82；评测（Phase 5 脚本）回报接近 0.85；时间记录 |

**模型利用（exploitation）监控**：每次评测同时记录"想象回报"（Eq. 2 的无折扣版本）与"仿真回报"，两者差距持续扩大即为利用信号，应对：缩短 H、增加世界模型更新步数、加大动作噪声。

### Phase 5：评测与消融（2–3 天）

| # | 任务 | 产出 | 验证 |
|---|---|---|---|
| 5.1 | `eval/benchmark.py`：每条 clip 从首帧整条播放（无 10 s 上限）；初始状态精确等于首帧（含关节/根速度），**关闭** D14 的重置噪声；摩擦 1.0、无 DR、无观测噪声、无动作噪声、确定性物理；失败 = 训练终止触发；归一化回报 = Σ_{t=1}^{T−1} r_t / (0.1 × (T − 1))，T 为 clip 帧数（从首帧出发有 T − 1 个奖励步，提前终止的 clip 分母不变） | 脚本 | 在 e2 检查点上输出 40 条逐 clip 结果 |
| 5.2 | `eval/push_test.py`：合成一条 20 s 静态 T-pose 参考 clip（关节角固定为 T-pose、根静止）作为 g_t；推力用 mjlab `push_by_setting_velocity` 的方式在根线速度上叠加 0.5 / 1 / 2 / 4 m/s（水平随机方向）；"稳定"判据（论文未给，自定）：根线速度范数 < 0.1 m/s 且根位置误差 < 0.05 m 持续 0.5 s，记录从推力到稳定的时间；摔倒 = 训练终止 | 脚本 | critic 有/无 对比 |
| 5.3 | e3：H ∈ {16, 32} × {critic, no critic}（4 组）。到达判据用**训练回报** 0.82（与论文 Fig. 4b/c 一致；训练回报 = 训练回合的归一化回报，分母为该回合实际长度 × 0.1，起点按 D14 采样）；每组另记 benchmark 评测回报；上限 8 h | Fig. 4b/c 对应 | 预期：无 critic H=16 失败 |
| 5.4 | 绘图脚本 | 图 | – |

### Phase 6：BONES-SEED 全量（数据到位后 1–2 周，含训练时间）

| # | 任务 | 产出 | 验证 |
|---|---|---|---|
| 6.1 | 全量转换（129.5k clip，CPU 并行；机器 B 128 核） | `library` 约 13.5 GB fp32 | 数量对照 |
| 6.2 | CPU pinned 库 + 按需 gather 路径；采集与想象的目标帧 gather 延迟测量 | – | gather 不成为瓶颈（< 5% 迭代时间） |
| 6.3 | DDP 训练入口（`torchrunx`，参考 `mjlab/scripts/train.py:182-227`；mjlab 传 `backend=None` 把进程组交给 runner，我们的 runner 要自己 `init_process_group("nccl")`）；4 卡预设，学习率缩放因子可配置，默认 √world_size（论文 2 卡 √2 的外推，4 卡即 2×）；动作库每 rank 一份会占 4 × 13.5 GB 锁页内存，改为 fp16 库（6.8 GB）直接常驻各卡显存，或多 rank 共享一份 pinned 内存 | 脚本 | 2 卡短跑 scaling 效率 ≥ 70% |
| 6.4 | e5：4×5090 全量训练，每 4000 迭代存检查点；benchmark 在训练结束后或在空闲卡上离线逐检查点运行（论文同样不把评测计入训练时间）。全量 benchmark 一次约 32 批、按 clip 长度分桶，估 15–30 min | 曲线、Table A.8 对应 | 失败数随检查点下降；回报曲线 |
| 6.5 | 数据长度扫描 1/7/40 h（Fig. 3b/c）——可选 | – | – |

### Phase 7（M6，可选）

- 速度接口（SPEC §16）：`interfaces/velocity.py`、阶跃测试。
- ONNX 导出与 sim2sim 验证（接入现有的 sim2sim 流程）。
- PPO-Sim 基线：mjlab PPO + 我们的环境配置，8192 环境，BeyondMimic 超参。
- SONIC ONNX 权重在我们 benchmark 上的评测。

---

## 7. 测试策略

- **数值对拍优先**：FK、奖励、观测头、状态回写四项必须与 MuJoCo/mjlab 逐元素对拍，这是整个方法的正确性基础。
- **梯度检查**：BPTT 目标对策略参数的有限差分检查在微型配置（H=3，B=2，宽 16）下进行。
- **每个 Phase 的 gate** 在第 6 节表格中，未通过不进入下一 Phase。
- **冒烟测试**进 CI 风格的 `make test-fast`（CPU 可跑的部分用 `FORCE_CPU=1`，参考 `mjlab/tests/conftest.py:22-30`）。

---

## 8. 实验矩阵与论文图表对照

| 论文图表 | 我们的实验 | 规模差异 |
|---|---|---|
| Fig. 4a（SuperMimic vs PPO-Sim，LAFAN1） | e2（+ M6 的 PPO-Sim） | 同数据；硬件 5090 |
| Fig. 4b/c（H 扫描 × critic） | e3 | 仅 H ∈ {16, 32}；H=64 在 32 GB 上需把想象批降到 2048 或做激活 checkpointing |
| Fig. 5a（世界模型 vs 仿真速度） | e4 `bench_wm_speed` | 5090 上倍数预计更低 |
| Fig. 5b（PPO-WM） | M6 可选 | – |
| Fig. 3a（回报 vs GPU 小时） | e5 | 5090 GPU 小时，单独标注 |
| Fig. 3c（预测误差 vs 数据量） | e5 + 6.5 | 归一化定义不同，只比趋势 |
| Table A.8（失败数） | e5 benchmark | 无 SONIC 列（除非 M6） |
| §4.3 推力测试 | 5.2 | 同协议 |

---

## 9. 计算预算与排期

**仿真吞吐校准**：以 Phase 0.3 实测为准。先验估计：5090 fp32 峰值高于 H100，MuJoCo Warp 在 4096 环境下有望 ≥ 1×10⁵ 步/s。

**显存（4096 想象 rollout × H=32，默认配置）估算**（按 autograd 实际保存量：Linear 保存输入、SiLU 保存预激活、tanh 保存输出）：

| 组件 | 估算 |
|---|---|
| BPTT 激活：动力学 ≈12.5k + 编码器 ≈8.4k + 解码器 ≈8.7k + canonicalize/integrate/FK 奖励 ≈2k ≈ 31–32k 浮点/步/rollout ≈ 125 KB × 32 × 4096 | ≈ 16.4 GB |
| allocator 碎片 10–15% | ≈ 2 GB |
| replay buffer 4096 × 1000 × 169 维 fp32（71+29+29+33+r+d+clip+frame+2 flags） | ≈ 2.8 GB |
| 世界模型拟合批 2048 × 8 | ≈ 1 GB |
| MuJoCo Warp 4096 世界（efc_J 250×35 + 接触/求解器工作区 ≈ 100–150 KB/世界）+ graphs | ≈ 0.5–1 GB |
| 参数 + 梯度 + Adam（18M） | ≈ 0.3 GB |
| CUDA 上下文 + warp 内核 | ≈ 1 GB |
| LAFAN1 动作库 | ≈ 0.1 GB |
| **峰值** | **≈ 25–27 GB** |

32 GB 放得下但余量小。H=64 的激活约 33 GB，4096 批下放不下。OOM 时第一步把想象批降到 2048（保持 H=32），第二步做逐步激活 checkpointing，第三步把 replay 放 CPU pinned。

**时间估算**（按 D12 的工作量拆分、5090 张量核约为 H100 的 1/4.7、fp32 仿真约 1.6×；想象侧若未做 3.6 的编译优化会受 launch 开销支配，故预留 50% 余量）：

| 实验 | 论文（H100） | 单卡 5090 估算 | 4×5090 估算 |
|---|---|---|---|
| e2 LAFAN1 到 0.82 | 1.2 h | 4–8 h | – |
| e3 四组消融 | – | 约 25 h 累计 | 并行 4 卡约 8 h |
| e5 超过 SONIC 回报水平 | 3 h | 10–20 h | 3–6 h |
| e5 完整 run（论文 1 GPU 曲线末端，Fig. 3a / A.1 横轴图读约 10 h，正文未给） | ≈10 h（图读） | 2–3 天 | 14–20 h |

**排期**（一人全职）：Phase 0–1 约 1 周；Phase 2 约 1.5–2 周（多 clip 命令项、回报采样器、StateIO、位置+速度 FK、5 个数值对拍测试）；Phase 3 约 1 周；Phase 4 约 1 周实现 + 2–3 周调参迭代（20 项自选决策 U1–U20 首个配置就达标的概率不高）；Phase 5 约 1 周；Phase 6 受数据到位与机器 B 档期影响，1–2 周。总计约 8–10 周到 M5。

---

## 10. 风险登记

| 编号 | 风险 | 影响 | 应对 |
|---|---|---|---|
| R1 | 机器 B 驱动 570 不满足 mjlab 要求 | 无法在机器 B 训练 | B1：升级驱动；备选 cu128 |
| R2 | BONES-SEED 许可审批慢 | M5 延期 | 先做 LAFAN1 全链路；M5 并行等待 |
| R3 | BPTT 梯度爆炸 / NaN | 训练失败 | 论文的双重裁剪；H 从 16 起步；TF32 关闭对照 |
| R4 | 世界模型被策略利用 | 仿真回报停滞而想象回报上升 | 4.6 的监控指标；缩短 H；增加 WM 更新 |
| R5 | 解析奖励头实现错误（FK/速度 FK 与 MuJoCo 不一致） | 策略优化错误目标 | 2.2/2.8 的机器精度对拍测试为 gate；D4 备选学习奖励头 |
| R6 | 显存不足 | 无法用默认批 | 第 9 节降批策略 |
| R7 | UTD 解释错误导致时间分配失衡 | 训练慢或模型欠拟合 | D12 的耗时分解监控；UTD 进配置 |
| R8 | 论文未指明项的决定偏离作者实现 | 数值不一致 | 所有决定可配置；EXPERIMENTS.md 记录对照 |
| R9 | mjlab 版本漂移：引用的是 main 的 8bb2bc9，PyPI 1.6.0 内容不同 | 构建失败或 API 不符 | git 依赖固定到 commit；升级单独评估 |
| R10 | 名字过滤关键词表与论文不同 | clip 数与论文不同 | 以 129,667 校准；±0.5% 外逐词排查 |
| R11 | reset 模式 DR 触发全局 `set_const` 重算 | 仿真吞吐骤降 | COM 用 startup；0.3 用真实 DR 配置测速 |
| R12 | 观测历史 term-major 布局在想象侧复制错误 | 策略在仿真与想象间不等价 | D20；对拍测试覆盖历史布局 |
| R13 | warp-lang / mujoco_warp 在 570 驱动 + cu128 下不可用 | 机器 B 不可用 | B1 的实测步骤；升级驱动 |
| R14 | 想象 rollout 受 kernel launch 支配 | 速度达不到"≥5× 仿真" | 3.6 编译/CUDA graph 为必做 |
| R15 | TF32 污染 FK 对拍测试 | 测试误报 | 测试内关闭 TF32；四元数逐元素实现 |
| R16 | BONES-SEED G1 CSV 的关节顺序/四元数顺序/fps 与 LAFAN1 不同 | 数据全错 | 1.4 硬门 |

---

## 11. 前置条件

1. 机器 B 的 NVIDIA 驱动 ≥ 580（mjlab main / torch cu130 的要求，见 B1）。
2. BONES-SEED 数据：在 Hugging Face `bones-studio/seed` 页面接受许可后，用可读的 HF token 下载 `g1.tar.gz`。
3. 日志后端：TensorBoard（W&B 不使用）。

---

## 附录 A：命令速查（实施后生效）

```bash
# 数据
uv run python scripts/data/download_lafan1.py --out data/raw/lafan1
uv run python scripts/data/csv_to_npz_batch.py --in data/raw/lafan1/g1 --out data/npz/lafan1 --input-fps 30
uv run python scripts/data/build_library.py --npz data/npz/lafan1 --out data/library/lafan1.npz
# 训练
uv run python scripts/train.py --preset lafan1 --num-gpus 1 --run-name e2_lafan1_40
uv run python scripts/train.py --preset bones_seed --num-gpus 4 --run-name e5_full
# 评测
uv run python scripts/eval_benchmark.py --ckpt logs/e2_lafan1_40/ckpt_latest.pt --library data/library/lafan1.npz
uv run python scripts/eval_push.py --ckpt ...
# 测试
make test-fast
```

## 附录 B：估算依据

- BPTT 激活（按 autograd 实际保存：Linear 保存输入、SiLU 保存预激活、tanh 保存输出）：动力学 133 + 6×2048 + 70 ≈ 12.5k；编码器 680 + 2×(2048+1024+512+256) + 32 ≈ 8.4k；解码器 962 + 7680 + 29 ≈ 8.7k；规范化/积分/FK 奖励中间量 ≈ 2k；合计 ≈ 31–32k 浮点 ≈ 125 KB/步/rollout；× 32 × 4096 ≈ 16.4 GB。与论文"H=128 在 80 GB 上刚好放下"（≈ 66 GB + 其他）量级自洽。
- 世界模型更新 FLOPs：10 × 2048 × 8 = 163,840 步 × 8.8M 参数 × 2 × 3（前向+反向）≈ 8.7 TFLOP。
- 策略更新 FLOPs：4096 × 32 = 131,072 步 × (8.8M × 2 × 2 + 9M × 2 × 3)（冻结的动力学网络只需输入梯度，不需权重梯度）≈ 4.6 + 7.1 ≈ 11.7 TFLOP。
- 以上 Fig. 5a 的仿真吞吐、Fig. 3a/A.1 的 10 h 均为对数坐标图读，不确定度约 ±2×，只用于量级判断，不作为验收阈值。
- 5090 TF32 张量核密集峰值约 105 TFLOPS；H100 SXM 约 495 TFLOPS。
