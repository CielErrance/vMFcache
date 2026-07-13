# vMFcache Baseline 分数记录

用于对照每次改动的效果。新增实验请在下方追加一节，不要覆盖历史记录。

**对照基准**：下文「Production baseline」为当前 `vMFcache.py` 在 test split、cache-only DQDA/κ、最优超参下的 FG10 全量结果。

---

## Production baseline（主文件对照）

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

### 超参（生产默认）

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

## 关键对照实验（摘要）

| 实验 | Average | 日志 ts | 备注 |
|------|---------|---------|------|
| **Production baseline（上表）** | **70.29** | `20260711_145411` | cache-κ，test split |
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
