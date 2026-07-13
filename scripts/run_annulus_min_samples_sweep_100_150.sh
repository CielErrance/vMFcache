#!/usr/bin/env bash
# annulus_min_samples sweep: 100 / 150 on FG10 test split (per-dataset parallel).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p scripts/logs

PY=/home/liangyiwen/miniconda3/envs/adapt/bin/python
ts=$(date +%Y%m%d_%H%M%S)
DATA_ROOT=/home/liangyiwen/datasets
GPUS=(0 1 2 4 5 6 7)
VALUES=(100 150)
DATASETS=(
  fgvc_aircraft caltech101 stanford_cars dtd eurosat
  oxford_flowers food101 oxford_pets sun397 ucf101
)

COMMON=(
  --data "${DATA_ROOT}"
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

META="scripts/logs/annulus_min_samples_sweep_100_150_${ts}.txt"
echo "annulus_min_samples sweep ts=${ts} values=100,150 baseline=200 test-split parallel=${#GPUS[@]}gpus" | tee "${META}"

run_one() {
  local gpu="$1"
  local n="$2"
  local ds="$3"
  local log="scripts/logs/sweep_annulus_min_n${n}_${ds}_gpu${gpu}_${ts}.log"
  echo "[launch] GPU${gpu} annulus_min_samples=${n} ${ds} -> ${log}" | tee -a "${META}"
  env PYTHONUNBUFFERED=1 WANDB_MODE=disabled "${PY}" ./vMFcache.py \
    "${COMMON[@]}" --gpu "${gpu}" --test_set "${ds}" --annulus_min_samples "${n}" \
    > "${log}" 2>&1
  local acc
  acc=$(grep -E "^${ds}:" "${log}" | tail -1 | awk '{print $2}' || echo "NA")
  echo "[done] GPU${gpu} annulus_min_samples=${n} ${ds} acc=${acc}" | tee -a "${META}"
}

: > "${META}.jobs"
max_jobs=${#GPUS[@]}
gpu_idx=0
job_idx=0

for n in "${VALUES[@]}"; do
  for ds in "${DATASETS[@]}"; do
    while (( $(jobs -r -p | wc -l) >= max_jobs )); do
      wait -n
    done
    gpu="${GPUS[$gpu_idx]}"
    gpu_idx=$(( (gpu_idx + 1) % max_jobs ))
    run_one "${gpu}" "${n}" "${ds}" &
    echo "${n} ${ds} ${gpu} $! ${ts}" >> "${META}.jobs"
    job_idx=$((job_idx + 1))
  done
done
wait

echo "=== Summary ===" | tee -a "${META}"
for n in "${VALUES[@]}"; do
  echo "--- annulus_min_samples=${n} ---" | tee -a "${META}"
  sum=0
  cnt=0
  for ds in "${DATASETS[@]}"; do
    log=$(ls scripts/logs/sweep_annulus_min_n${n}_${ds}_gpu*_${ts}.log 2>/dev/null | head -1)
    acc="NA"
    if [[ -f "${log}" ]]; then
      acc=$(grep -E "^${ds}:" "${log}" | tail -1 | awk '{print $2}' || true)
    fi
    echo "${ds}: ${acc}" | tee -a "${META}"
    if [[ -n "${acc}" && "${acc}" != "NA" ]]; then
      sum=$(python3 -c "print(${sum}+float('${acc}'))")
      cnt=$((cnt + 1))
    fi
  done
  if (( cnt > 0 )); then
    avg=$(python3 -c "print(round(${sum}/${cnt}, 2))")
    echo "Average: ${avg}" | tee -a "${META}"
  fi
done
echo "done -> ${META}"
echo "  bash scripts/parse_hp_sweep_results.sh annulus_min_samples ${ts}"
