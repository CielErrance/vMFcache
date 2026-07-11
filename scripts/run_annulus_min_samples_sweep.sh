#!/usr/bin/env bash
# annulus_min_samples sweep on FG10 (optimal hyperparams otherwise).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p scripts/logs

PY=/home/liangyiwen/miniconda3/envs/adapt/bin/python
ts=$(date +%Y%m%d_%H%M%S)
DATA_ROOT=/home/liangyiwen/datasets
TESTSETS=fgvc_aircraft/caltech101/stanford_cars/dtd/eurosat/oxford_flowers/food101/oxford_pets/sun397/ucf101
GPUS=(0 1 2 4 5 6 7)
# baseline annulus_min_samples=200
VALUES=(50 100 300)

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
  --rho 2.0
  --chi2_low 0.05
  --chi2_high 0.95
  --lambda_div 1.0
  --clip_weight 1.0
)

META="scripts/logs/annulus_min_samples_sweep_${ts}.txt"
echo "annulus_min_samples sweep ts=${ts} baseline=200 ps_temperature=40 eta=0.75 rho=2" | tee "${META}"

launch_one() {
  local gpu="$1"
  local n="$2"
  local log="scripts/logs/sweep_annulus_min_n${n}_gpu${gpu}_${ts}.log"
  echo "[launch] GPU${gpu} annulus_min_samples=${n} -> ${log}" | tee -a "${META}"
  env PYTHONUNBUFFERED=1 WANDB_MODE=disabled "${PY}" ./vMFcache.py \
    "${COMMON[@]}" --gpu "${gpu}" --annulus_min_samples "${n}" \
    > "${log}" 2>&1
  local avg
  avg=$(grep -E "^Average:" "${log}" | tail -1 | awk '{print $2}' || echo "NA")
  echo "[done] GPU${gpu} annulus_min_samples=${n} avg=${avg}" | tee -a "${META}"
}

: > "${META}.jobs"
idx=0
for n in "${VALUES[@]}"; do
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  launch_one "${gpu}" "${n}" &
  echo "${n} ${gpu} $! ${ts}" >> "${META}.jobs"
  idx=$((idx + 1))
done
wait

echo "done -> ${META}"
echo "  bash scripts/parse_hp_sweep_results.sh annulus_min_samples ${ts}"
