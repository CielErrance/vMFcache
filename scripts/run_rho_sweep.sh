#!/usr/bin/env bash
# rho sweep on FG10 (eta=0.75, ps_temperature=40 fixed).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p scripts/logs

PY=/home/liangyiwen/miniconda3/envs/adapt/bin/python
ts=$(date +%Y%m%d_%H%M%S)
DATA_ROOT=/home/liangyiwen/datasets
TESTSETS=fgvc_aircraft/caltech101/stanford_cars/dtd/eurosat/oxford_flowers/food101/oxford_pets/sun397/ucf101
GPUS=(0 1 2 4 5)
# baseline rho=2.0
VALUES=(0.5 1 2 3 4)

COMMON=(
  --data "${DATA_ROOT}"
  --test_set "${TESTSETS}"
  --arch ViT-B/16
  --bank_size 16
  --alpha 0.9
  --class_type Custom
  --GPT
  --var_aligned_kappa
  --ps_temperature 40
  --eta 0.75
  --chi2_low 0.05
  --chi2_high 0.95
  --annulus_min_samples 200
  --div_floor 0.5
  --clip_weight 1.0
)

META="scripts/logs/rho_sweep_${ts}.txt"
echo "rho sweep ts=${ts} baseline=2.0 eta=0.75 ps_temperature=40" | tee "${META}"

launch_one() {
  local gpu="$1"
  local rho="$2"
  local tag="${rho//./p}"
  local log="scripts/logs/sweep_rho_r${tag}_gpu${gpu}_${ts}.log"
  echo "[launch] GPU${gpu} rho=${rho} -> ${log}" | tee -a "${META}"
  env PYTHONUNBUFFERED=1 WANDB_MODE=disabled "${PY}" ./vMFcache.py \
    "${COMMON[@]}" --gpu "${gpu}" --rho "${rho}" \
    > "${log}" 2>&1
  local avg
  avg=$(grep -E "^Average:" "${log}" | tail -1 | awk '{print $2}' || echo "NA")
  echo "[done] GPU${gpu} rho=${rho} avg=${avg}" | tee -a "${META}"
}

: > "${META}.jobs"
idx=0
for rho in "${VALUES[@]}"; do
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  launch_one "${gpu}" "${rho}" &
  echo "${rho} ${gpu} $! ${ts}" >> "${META}.jobs"
  idx=$((idx + 1))
done
wait

echo "done -> ${META}"
echo "  bash scripts/parse_hp_sweep_results.sh rho ${ts}"
