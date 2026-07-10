#!/usr/bin/env bash
# Example: eurosat smoke run for vMFcache
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/liangyiwen/miniconda3/envs/adapt/bin/python
GPU="${1:-2}"

env PYTHONUNBUFFERED=1 WANDB_MODE=disabled "${PY}" ./vMFcache.py \
  --data /home/liangyiwen/datasets \
  --test_set eurosat \
  --arch ViT-B/16 \
  --bank_size 16 \
  --alpha 0.9 \
  --class_type Custom \
  --GPT \
  --gpu "${GPU}" \
  --var_aligned_kappa \
  --ps_temperature 175 \
  --eta 0.75 \
  --rho 2.0 \
  --chi2_low 0.05 \
  --chi2_high 0.95 \
  --annulus_min_samples 200 \
  --div_floor 0.5 \
  --clip_weight 1.0
