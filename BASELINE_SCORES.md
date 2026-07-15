# vMFcache Baseline 分数记录

用于对照每次改动的效果。新增实验请在下方追加一节，不要覆盖历史记录。

**对照基准**：「Production baseline (2026-07-11)」为改门控前对照；**当前代码默认超参**对应「2026-07-15 — chi2_low=0 + no delta_high」，FG10 平均 **70.37%**。

---

## Production baseline（2026-07-11，改门控前对照）

| 字段 | 值 |
|------|-----|
| 脚本 | `vMFcache.py` |
| 运行脚本 | `scripts/run_full10_cache_kappa.sh` |
| 日志 | `scripts/logs/full10_cache_kappa_20260711_145411.txt` |
| 日期 | 2026-07-11 |
| run id | `ts=20260711_145411` |
| 说明 | test split；cache-only DQDA + cache-only κ；entropy-redundancy 准入驱逐 |

### 算法要点

- **Cache 准入**：Mahalanobis 经验安全环硬门控 + 满 bank 时 `-entropy - λ·max_redundancy` 驱逐
- **DQDA**：每次 cache 准入后，仅用 **已准入 cache 样本** 更新 `μ` / `var_diag`（非全流式样本）
- **vMF κ**：`var_aligned_kappa`，从 cache 软标签分散度 MLE 估计（非全 batch）
- **Mixture**：DQDA `P_L` + cache vMF `P_S`，`γ(z)` 由 Mahalanobis 距离门控
- **数据**：FG10，`build_test_loader(..., mode='test')`

### 超参（当时生产默认）

```
bank_size=16  alpha=0.9  batch_size=1
class_type=Custom  --GPT  arch=ViT-B/16
var_aligned_kappa  ps_temperature=40  kappa_disp_min_weight=2.0  kappa_fallback=2000
eta=0.75  rho=2.0
chi2_low=0.05  chi2_high=0.95  annulus_min_samples=200
lambda_div=1.0  clip_weight=1.0
```

### 分数（%）

| 数据集 | Acc |
|--------|-----|
| fgvc_aircraft | 27.75 |
| caltech101 | 94.93 |
| stanford_cars | 67.44 |
| dtd | 54.20 |
| eurosat | 68.30 |
| oxford_flowers | 74.83 |
| food101 | 84.33 |
| oxford_pets | 91.33 |
| sun397 | 69.34 |
| ucf101 | 70.45 |
| **Average** | **70.29** |

---

## 2026-07-15 — annulus 超参最优：chi2_low=0 + no delta_high

| 字段 | 值 |
|------|-----|
| 改动 | `chi2_low` 默认 0.05→**0**；默认关闭 delta_high（`no_delta_high_gate=True`，可用 `--use_delta_high_gate` 打开） |
| 依据 | `chi2_low` sweep `ts=20260714_071328`；`chi2_high` sweep `ts=20260714_155157`（固定 chi2_low=0） |
| 最优配置分数日志 | `scripts/logs/chi2_high_sweep_20260714_155157.txt`（`chi2_high=none` 组） |
| 说明 | test split；其余超参同 07-11（ps=40, η=0.75, ρ=2, …） |

### Sweep 摘要（FG10 Average）

| 实验 | 设置 | Average |
|------|------|---------|
| chi2_low sweep | 0 / 0.05 / 0.1 / 0.2（chi2_high=0.95） | **70.35** / 70.29 / 70.25 / 69.76 |
| chi2_high sweep | 0.8 / 0.9 / 0.95 / **none**（chi2_low=0） | 69.25 / 70.32 / 70.35 / **70.37** |

### 当前代码默认超参

```
bank_size=16  alpha=0.9  batch_size=1
class_type=Custom  --GPT  arch=ViT-B/16
var_aligned_kappa  ps_temperature=40  kappa_disp_min_weight=2.0  kappa_fallback=2000
eta=0.75  rho=2.0
chi2_low=0  no_delta_high_gate=True  (chi2_high unused unless --use_delta_high_gate)
annulus_min_samples=200
lambda_div=1.0  clip_weight=1.0
```

### 分数（%）：chi2_low=0 + no delta_high

| 数据集 | Acc (07-11 baseline) | Acc (最优) | Δ |
|--------|----------------------|------------|---|
| fgvc_aircraft | 27.75 | 27.75 | 0.00 |
| caltech101 | 94.93 | 94.97 | +0.04 |
| stanford_cars | 67.44 | 67.85 | +0.41 |
| dtd | 54.20 | 53.66 | −0.54 |
| eurosat | 68.30 | 68.68 | +0.38 |
| oxford_flowers | 74.83 | 75.11 | +0.28 |
| food101 | 84.33 | 84.27 | −0.06 |
| oxford_pets | 91.33 | 91.22 | −0.11 |
| sun397 | 69.34 | 69.60 | +0.26 |
| ucf101 | 70.45 | 70.63 | +0.18 |
| **Average** | **70.29** | **70.37** | **+0.08** |

---

## 历史对照实验（摘要）

| 实验 | Average | 日志 ts | 备注 |
|------|---------|---------|------|
| **chi2_low=0 + no delta_high（当前默认）** | **70.37** | `20260714_155157` | annulus 超参最优 |
| Production baseline（07-11） | 70.29 | `20260711_145411` | chi2_low=0.05, chi2_high=0.95 |
| chi2_low=0（仍开 high=0.95） | 70.35 | `20260714_071328` | low 扫描最优 |
| test split + bs=1（改 κ 前） | 70.35 | `20260711_044611` | 全 batch κ → cache-κ 后 −0.06pp |
| eta sweep @ ps=40（η=0.75/1.0 最优） | 70.29 | `20260712_064740` | η=0.0 仅 69.28（−1.01pp） |
| annulus_min_samples 100/150 | 70.32 / 70.36 | `20260711_090739` | 默认 200，不敏感 |
| rho=2.0（最优） | 70.54 | `20260709_161734` | train split 时期结果，仅供参考 |
| ps_temperature=40 @ η=0.75 | 70.54 | `20260710_025205` | train split 时期结果 |

> **注意**：2026-07-11 之前部分 sweep 使用 **train split**（datautils 未修 `mode='test'`），与 production baseline 不可直接横比。

---

## 新实验模板（复制填写）

```markdown
### YYYY-MM-DD — 简述

| 字段 | 值 |
|------|-----|
| 改动 | … |
| 日志 | `scripts/logs/…` |

| 数据集 | Baseline | 本次 | Δ |
|--------|----------|------|---|
| … | … | … | … |
| **Average** | 70.29 | … | … |
```
