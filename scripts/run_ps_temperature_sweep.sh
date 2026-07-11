#!/usr/bin/env bash
# ps_temperature sweep on FG10 (eta=0.75 fixed, var_aligned_kappa).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p scripts/logs

PY=/home/liangyiwen/miniconda3/envs/adapt/bin/python
ts=$(date +%Y%m%d_%H%M%S)
DATA_ROOT=/home/liangyiwen/datasets
TESTSETS=fgvc_aircraft/caltech101/stanford_cars/dtd/eurosat/oxford_flowers/food101/oxford_pets/sun397/ucf101
GPUS=(0 1 2 4 5 6 7)
# baseline ps_temperature=175
VALUES=(29 40 50 75 100 125 150 175 200)

COMMON=(
  --data "${DATA_ROOT}"
  --test_set "${TESTSETS}"
  --arch ViT-B/16
  --bank_size 16
  --alpha 0.9
  --class_type Custom
  --GPT
  --var_aligned_kappa
  --eta 0.75
  --rho 2.0
  --chi2_low 0.05
  --chi2_high 0.95
  --annulus_min_samples 200
  --clip_weight 1.0
)

META="scripts/logs/ps_temperature_sweep_${ts}.txt"
echo "ps_temperature sweep ts=${ts} baseline=175 eta=0.75" | tee "${META}"

launch_one() {
  local gpu="$1"
  local temp="$2"
  local tag="${temp//./p}"
  local log="scripts/logs/sweep_ps_temp_t${tag}_gpu${gpu}_${ts}.log"
  echo "[launch] GPU${gpu} ps_temperature=${temp} -> ${log}" | tee -a "${META}"
  env PYTHONUNBUFFERED=1 WANDB_MODE=disabled "${PY}" ./vMFcache.py \
    "${COMMON[@]}" --gpu "${gpu}" --ps_temperature "${temp}" \
    > "${log}" 2>&1
  local avg
  avg=$(grep -E "^Average:" "${log}" | tail -1 | awk '{print $2}' || echo "NA")
  echo "[done] GPU${gpu} ps_temperature=${temp} avg=${avg}" | tee -a "${META}"
}

: > "${META}.jobs"
idx=0
for temp in "${VALUES[@]}"; do
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  launch_one "${gpu}" "${temp}" &
  echo "${temp} ${gpu} $! ${ts}" >> "${META}.jobs"
  idx=$((idx + 1))
  if (( idx % ${#GPUS[@]} == 0 )); then
    echo "[wave] waiting for batch of ${#GPUS[@]} jobs..." | tee -a "${META}"
    wait
    echo "[wave] batch done" | tee -a "${META}"
  fi
done
remaining=$((idx % ${#GPUS[@]}))
if (( remaining > 0 )); then
  echo "[wave] waiting for final batch of ${remaining} jobs..." | tee -a "${META}"
  wait
fi

echo "done -> ${META}"
echo "  bash scripts/parse_hp_sweep_results.sh ps_temperature ${ts}"
