#!/usr/bin/env bash
# eta sweep on FG10 (ps_temperature=40, rho=2 fixed).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p scripts/logs

PY=/home/liangyiwen/miniconda3/envs/adapt/bin/python
ts=$(date +%Y%m%d_%H%M%S)
DATA_ROOT=/home/liangyiwen/datasets
TESTSETS=fgvc_aircraft/caltech101/stanford_cars/dtd/eurosat/oxford_flowers/food101/oxford_pets/sun397/ucf101
GPUS=(0 1 2 4 5 6 7)
# baseline eta=0.75
VALUES=(0.0 0.25 0.5 0.75 1.0 1.25 1.5)

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
  --rho 2.0
  --chi2_low 0.05
  --chi2_high 0.95
  --annulus_min_samples 200
  --clip_weight 1.0
)

META="scripts/logs/eta_sweep_ps40_${ts}.txt"
echo "eta sweep ts=${ts} baseline=0.75 ps_temperature=40 rho=2" | tee "${META}"

launch_one() {
  local gpu="$1"
  local eta="$2"
  local tag="${eta//./p}"
  local log="scripts/logs/sweep_eta_ps40_e${tag}_gpu${gpu}_${ts}.log"
  echo "[launch] GPU${gpu} eta=${eta} -> ${log}" | tee -a "${META}"
  env PYTHONUNBUFFERED=1 WANDB_MODE=disabled "${PY}" ./vMFcache.py \
    "${COMMON[@]}" --gpu "${gpu}" --eta "${eta}" \
    > "${log}" 2>&1
  local avg
  avg=$(grep -E "^Average:" "${log}" | tail -1 | awk '{print $2}' || echo "NA")
  echo "[done] GPU${gpu} eta=${eta} avg=${avg}" | tee -a "${META}"
}

: > "${META}.jobs"
idx=0
for eta in "${VALUES[@]}"; do
  gpu="${GPUS[$((idx % ${#GPUS[@]}))]}"
  launch_one "${gpu}" "${eta}" &
  echo "${eta} ${gpu} $! ${ts}" >> "${META}.jobs"
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
echo "  bash scripts/parse_hp_sweep_results.sh eta_ps40 ${ts}"
