# SuperMimic 复现记录（RTX 5080 16GB）

本文件记录在本机（1×RTX 5080 16GB）上复现 **SuperMimic: Scaling Humanoid
Whole-Body Control with One Differentiable World Model**（arXiv:2610.28480，
CoRL 2026 oral）的过程、配置、结果与偏离项。

代码底座：[`CuboiLeo/SuperMimic-Reproduced`](https://github.com/CuboiLeo/SuperMimic-Reproduced)
（独立复现，MIT），固定 mjlab commit `8bb2bc9`。

> 目标不是刷到论文的 129,514 条 BONES-SEED，而是在 **LAFAN1 40 条 / 2.45 h**
> 上复现论文 Fig. 4a 的量级（单卡 1.2 h → 评测回报 0.82），并记录
> 16GB 显卡上的可行配置。

---

## 1. 硬件与环境

| 项 | 值 |
|---|---|
| GPU | NVIDIA RTX 5080，16 GB，sm_120 |
| CPU | 16 核 |
| 内存 | 46 GB |
| 系统 | Ubuntu 24.04 |
| Python | 3.12（repo `.python-version`） |
| PyTorch | 2.14.1+cu130 |
| MuJoCo | 3.16 |
| MuJoCo Warp | 3.16 |
| warp-lang | 1.18 |
| mjlab | commit `8bb2bc9`（`uv` 从 GitHub 拉取） |

环境创建：

```bash
cd supermimic-repro
env -u PYTHONPATH uv sync --group dev
env -u PYTHONPATH uv run pytest tests -q
```

本机实测：**26 passed**（含 LAFAN1 数据、GPU、FK、解析头、想象与评测回归测试）。

---

## 2. 数据

LAFAN1 G1 重定向 CSV（40 条，30 fps）来自 HuggingFace
`lvhaidong/LAFAN1_Retargeting_Dataset` 的 `g1/`。由于本机 `huggingface.co`
DNS 不通，使用镜像 `HF_ENDPOINT=https://hf-mirror.com` 下载。

```bash
HF_ENDPOINT=https://hf-mirror.com .venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("lvhaidong/LAFAN1_Retargeting_Dataset",
                  repo_type="dataset", allow_patterns=["g1/*.csv"],
                  local_dir="data/raw/lafan1", max_workers=8)
PY
```

转换与打包：

```bash
env -u PYTHONPATH uv run python scripts/data/csv_to_npz_batch.py \
  --input-dir data/raw/lafan1/g1 --output-dir data/npz/lafan1 \
  --input-fps 30 --workers 16
env -u PYTHONPATH uv run python scripts/data/build_library.py \
  --npz-dir data/npz/lafan1 --out data/library/lafan1.npz
```

实测：

- 40 条 CSV → 40 条 50 Hz npz，**2.451 h**（论文 2.45 h ✓）
- 库文件 60 MB，441,140 帧
- 转换 3 s（16 workers）

---

## 3. 冒烟测试与显存

### 3.1 1024 envs / batch 1024，无 `torch.compile`

```bash
PYTHONPATH= .venv/bin/python -m supermimic.scripts.train \
  --run-name smoke1 --data.clips walk1_subject1 \
  --env.num-envs 1024 --policy-trainer.batch-size 1024 \
  --wm-trainer.batch-size 1024 \
  --policy-trainer.compile None --wm-trainer.compile None \
  --schedule.total-iterations 500 --schedule.max-hours 0.1
```

- **1.5 it/s**
- 每更新：sim 0.3 s / WM 2.6 s / policy **10 s**（Python FK 的 kernel launch 开销，未 compile）
- 显存 < 7 GB

### 3.2 1024 envs / batch 1024，`torch.compile=default`

```bash
PYTHONPATH= .venv/bin/python -m supermimic.scripts.train \
  --run-name smoke2 --data.clips walk1_subject1 \
  --env.num-envs 1024 --policy-trainer.batch-size 1024 \
  --wm-trainer.batch-size 1024 \
  --policy-trainer.compile default --wm-trainer.compile default \
  --schedule.total-iterations 300 --schedule.max-hours 0.15
```

- 首次 compile ~3 min；之后 **6.2 it/s**
- 每更新：sim 0.3 s / WM 1.2 s / policy 1.7 s
- 显存峰值 **6.5 GB**
- 300 迭代：WM nse1 0.039，nse8 0.143，想象存活率 0.29（早期）

### 3.3 2048 envs / batch 2048，`torch.compile=default`

```bash
PYTHONPATH= .venv/bin/python -m supermimic.scripts.train \
  --run-name smoke3 --data.clips walk1_subject1 \
  --env.num-envs 2048 --policy-trainer.batch-size 2048 \
  --wm-trainer.batch-size 2048 \
  --policy-trainer.compile default --wm-trainer.compile default \
  --schedule.total-iterations 600 --schedule.max-hours 0.2
```

- 首次 compile ~3 min；之后 **4.3 it/s**
- 每更新：sim 0.3 s / WM 2.2 s / policy 2.1 s
- 显存峰值 **11.3 GB**（16GB 内，留 ~4.7GB 余量）
- 600 迭代：WM nse1 **0.030**，nse8 **0.108**，想象存活率 **0.59**

**结论**：2048/2048 是 16GB 上的稳定配置。4096/4096 会 OOM；
4096/2048 理论吞吐略高但显存逼近上限，长跑风险不值得。

> 正式 run：`logs/lafan1_anchor`，2048/2048，anchor obs，H=32，UTD=0.4，
> 总迭代 320k（保持论文 cosine 调度），`max-hours=3.3`。预计 ~51k 迭代。

---

## 4. 正式 run 计划

### Run A：论文观测口径（`anchor_obs=False`）

- 数据：LAFAN1 40 条
- `num_envs=2048`，`policy batch=2048`，`wm batch=2048`
- H=32，UTD=0.4，其余论文默认
- 目标：24k–32k 迭代（约 2–3 h）
- 预期：参照独立复现的 e2/e7，**评测回报 ~0.55–0.58**；原因是纯本体观测
  没有根部相对位姿，整条 220 s 的 LAFAN1 clip 会系统性漂移，根部两项奖励
  几乎拿不到。

### Run B：+ 锚点相对观测（`anchor_obs=True`，偏离论文 Table A.1）

- 同上，但 `goal` 追加 9 维锚点相对位姿
- 目标：24k 迭代（0.82 量级）→ 48k 迭代（0.90 量级）
- 预期：参照 e3，24k 迭代 **0.824**，48k 迭代 **0.904**
- 这是“把论文 0.82 复现出来”的主结果；必须明确标注它偏离了论文的
  纯本体观测设定。

### 评测（Appendix F）

```bash
PYTHONPATH= .venv/bin/python scripts/eval_benchmark.py \
  --ckpt logs/<run>/ckpt_latest.pt --npz-dir data/npz/lafan1
```

协议：每条 clip 从首帧整条确定性播放，无噪声、无 reset noise、无域随机化，
足底摩擦 1.0；终止触发即失败；归一化回报 = Σ奖励 / (0.1 × (T−1))，
其中 T 为 clip 帧数；提前终止不缩短分母，也不计入第 T 个额外奖励步。

---

## 5. 结果

正式 run：`logs/lafan1_anchor`（训练日志/checkpoint 不随仓库分发），LAFAN1 40 条，
2048 envs / batch 2048，anchor obs，H=32，UTD=0.4，3.3 h 上限，实际 48,200 迭代。
发布的结果在 `results/lafan1_anchor/`。

以下均为 2026-10-26 修正全长分母和额外终端奖励步后的重评结果；旧 JSON 保留作历史记录，不再用于结论。

| 检查点 | 训练时长 | 评测回报（mean） | step-weighted | 摔倒 | 评测次数 |
|---|---:|---:|---:|---:|---:|
| 24k | ~1.6 h | 0.7177 | 0.7320 | 9/40 | 1 |
| 40k | ~2.7 h | **0.8515** | **0.8608** | 2/40 | 1 |
| 44k | ~3.0 h | 0.8510 | 0.8563 | 3/40 | 1 |
| 48k | 3.3 h | 0.8375（0.8343–0.8403） | 0.8504 | 4–6/40 | 3 |

对照：

| 来源 | 硬件/配置 | 评测回报 | 摔倒 |
|---|---|---:|---:|
| 论文 Fig. 4a | 1×H100，4096 envs | **0.82** | – |
| 独立复现 e3 24k | 1×5090，4096/4096 | 0.824 | 5/40 |
| 独立复现 e3 48k | 1×5090，4096/4096 | 0.904 | 3/40 |
| **本机 40k** | **1×5080，2048/2048** | **0.8515（单次）** | **2/40** |
| **本机 48k** | **1×5080，2048/2048** | **0.8375（三次均值）** | **4–6/40** |

这些结果使用额外的 anchor 观测，且 env/batch、硬件和评测实现与论文及参考复现不同，不能据此宣称严格超过论文。当前单次结果中 40k 高于 48k，但是否为稳定结论必须靠多训练 seed 验证；不能把差异直接归因于 batch/env 减半。

48k 三次评测有 3/40 条 clip 的轨迹或失败状态发生变化，说明当前 GPU 仿真评测并非逐位确定；因此报告区间而非单次失败名单。

世界模型：最终 1 步误差 **0.0086**，8 步误差 **0.017**，想象存活率 **1.000**。
（论文 0.007–0.011 是 BONES-SEED 全量/策略在线口径，LAFAN1 上不能直接比。）

视频：

- `results/lafan1_anchor/videos/dance1_subject1.mp4`（600 步，12 s）
- `results/lafan1_anchor/videos/walk1_subject1.mp4`（800 步，16 s）
- `results/lafan1_anchor/videos/dance_sheet.png` / `walk_sheet.png`（四帧拼图）

### Run A（论文原始 93 维纯本体观测）

尚未跑。参考复现的 e2/e7 报告纯本体观测约 **0.556–0.582**，但不能替代本机对照。
匹配实验固定 48k 迭代与原始 LR 曲线，对两种观测各跑 3 个训练 seed：

```bash
for seed in 0 1 2; do
  bash scripts/run_lafan1.sh "lafan1_paper_s${seed}" 0 2048 2048 3.5 48000 "$seed"
  bash scripts/run_lafan1.sh "lafan1_anchor_s${seed}" 1 2048 2048 3.5 48000 "$seed"
done
```

4096/4096 在本机 16GB 上 OOM，因此该矩阵能对齐观测和迭代预算，不能声称与
5090 参考复现完全同配置；batch/env 的影响需另做缩容对照。

---

## 6. 已知偏离与解释

1. **显存 16GB vs 论文/参考 32GB**：环境数与批大小从 4096 降到 2048
   （或 1024）。这会改变有效数据吞吐，但算法与超参比例不变。
2. **`anchor_obs=True` 偏离论文 Table A.1**：论文的策略观测只有 93 维本体
   项，没有锚点相对位姿。独立复现发现，纯本体观测在 LAFAN1 整条 clip 上
   会漂移，评测回报封顶 ~0.56；加 9 维锚点相对位姿后到 0.82–0.90。
   这提示论文可能使用了未写明的状态估计，或者 LAFAN1 评测不是整条播放
   （例如分段/重置）；在本机 Run A 完成前，这些仍是待验证假设。
3. **LAFAN1 评测协议未写明**：论文只说“每个检查点在全部 40 条 clip 上
   打分”，没有说是否整条播放。BONES-SEED 的 clip 中位长度 5.8 s，
   漂移影响小；LAFAN1 平均 220 s，漂移影响大。因此论文的 0.82 与
   BONES-SEED 的 0.90 可能不是同一评测口径。
4. **未做**：PPO-Sim/PPO-WM 基线、H×critic 消融、ONNX、真机部署。
   这些是论文 Fig. 4/5 的对照，不影响主复现。

---

## 7. 复现结论

- [x] 修正评测后，Run B 40k 单次为 **0.8515 / 0.8608，2/40 摔倒**。
- [x] Run B 48k 三次均值为 **0.8375 / 0.8504**，失败 4–6/40；不能宣称 48k 最优。
- [x] 世界模型最终 1 步误差 0.0086，8 步 0.017；想象存活率 1.000。
- [x] dance1 / walk1 视频渲染正常，参考 ghost 与策略基本重合。
- [ ] Run A（论文原始 93 维纯本体观测）和 Run B 多训练 seed 尚未跑；当前不能
  分离观测、batch/env 与随机种子的影响。

**一句话结论**：在 1×RTX 5080 16GB、anchor-relative 观测下，方法能够训练并得到
约 0.84–0.85 的整 clip 回报；这支持“方法可运行且达到相近量级”，但不是与论文
同协议的严格复现，也不支持“48k 超过论文”的旧结论。
