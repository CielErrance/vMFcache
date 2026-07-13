#!/usr/bin/env bash
# FG10 eval: cache-based vMF kappa (test split, optimal hyperparams).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p scripts/logs

PY=/home/liangyiwen/miniconda3/envs/adapt/bin/python
ts=$(date +%Y%m%d_%H%M%S)
META="scripts/logs/full10_cache_kappa_${ts}.txt"
: > "${META}"

GPUS=(0 1 2 4 5 6 7)
DATASETS=(
  fgvc_aircraft caltech101 stanford_cars dtd eurosat
  oxford_flowers food101 oxford_pets sun397 ucf101
)

COMMON=(
  --data /home/liangyiwen/datasets
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
  --annulus_min_samples 200
  --lambda_div 1.0
  --clip_weight 1.0
)

echo "FG10 cache-kappa ts=${ts} test-split ps_temperature=40 eta=0.75 rho=2 batch_size=1" | tee -a "${META}"

run_one() {
  local gpu="$1"
  local ds="$2"
  local log="scripts/logs/cache_kappa_${ds}_gpu${gpu}_${ts}.log"
  echo "[launch] GPU${gpu} ${ds} -> ${log}" | tee -a "${META}"
  env PYTHONUNBUFFERED=1 WANDB_MODE=disabled "${PY}" ./vMFcache.py \
    "${COMMON[@]}" \
    --gpu "${gpu}" \
    --test_set "${ds}" \
    > "${log}" 2>&1
  local acc
  acc=$(grep -E "^${ds}:" "${log}" | tail -1 | awk '{print $2}' || echo "NA")
  echo "[done] GPU${gpu} ${ds} acc=${acc}" | tee -a "${META}"
}

max_jobs=${#GPUS[@]}
gpu_idx=0

for ds in "${DATASETS[@]}"; do
  while (( $(jobs -r -p | wc -l) >= max_jobs )); do
    wait -n
  done
  gpu="${GPUS[$gpu_idx]}"
  gpu_idx=$(( (gpu_idx + 1) % max_jobs ))
  run_one "${gpu}" "${ds}" &
  echo "${ds} ${gpu} $! ${ts}" >> "${META}.jobs"
done

wait
echo "=== Summary ===" | tee -a "${META}"
sum=0
n=0
for ds in "${DATASETS[@]}"; do
  log=$(ls scripts/logs/cache_kappa_${ds}_gpu*_${ts}.log 2>/dev/null | head -1)
  acc="NA"
  if [[ -f "${log}" ]]; then
    acc=$(grep -E "^${ds}:" "${log}" | tail -1 | awk '{print $2}' || true)
  fi
  echo "${ds}: ${acc}" | tee -a "${META}"
  if [[ -n "${acc}" && "${acc}" != "NA" ]]; then
    sum=$(python3 -c "print(${sum}+float('${acc}'))")
    n=$((n + 1))
  fi
done
if (( n > 0 )); then
  avg=$(python3 -c "print(round(${sum}/${n}, 2))")
  echo "Average: ${avg}" | tee -a "${META}"
fi
echo "meta=${META}"
