#!/usr/bin/env bash
# chi2_low (empirical delta_low quantile) sweep on FG10 test split.
# Searches: 0, 0.05, 0.1, 0.2  on GPUs 4,5,6,7.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p scripts/logs

PY=/home/liangyiwen/miniconda3/envs/adapt/bin/python
ts=$(date +%Y%m%d_%H%M%S)
DATA_ROOT=/home/liangyiwen/datasets
GPUS=(4 5 6 7)
VALUES=(0 0.05 0.1 0.2)
DATASETS=(
  fgvc_aircraft caltech101 stanford_cars dtd eurosat
  oxford_flowers food101 oxford_pets sun397 ucf101
)

COMMON=(
  --data "${DATA_ROOT}"
  --arch ViT-B/16
  --bank_size 16
  --alpha 0.9
  --batch_size 1
  --class_type Custom
  --GPT
  --var_aligned_kappa
  --ps_temperature 40
  --eta 0.75
  --rho 2.0
  --chi2_high 0.95
  --annulus_min_samples 200
  --lambda_div 1.0
  --clip_weight 1.0
)

META="scripts/logs/chi2_low_sweep_${ts}.txt"
echo "chi2_low/delta_low quantile sweep ts=${ts} values=${VALUES[*]} GPUs=${GPUS[*]} baseline chi2_low=0.05" | tee "${META}"

run_one() {
  local gpu="$1"
  local qlow="$2"
  local ds="$3"
  local tag="${qlow//./p}"
  local log="scripts/logs/sweep_chi2_low_q${tag}_${ds}_gpu${gpu}_${ts}.log"
  echo "[launch] GPU${gpu} chi2_low=${qlow} ${ds} -> ${log}" | tee -a "${META}"
  if nohup env PYTHONUNBUFFERED=1 WANDB_MODE=disabled "${PY}" ./vMFcache.py \
    "${COMMON[@]}" --gpu "${gpu}" --test_set "${ds}" --chi2_low "${qlow}" \
    > "${log}" 2>&1; then
    local acc
    acc=$(grep -E "^${ds}:" "${log}" | tail -1 | awk '{print $2}' || echo "NA")
    echo "[done] GPU${gpu} chi2_low=${qlow} ${ds} acc=${acc}" | tee -a "${META}"
  else
    echo "[FAIL] GPU${gpu} chi2_low=${qlow} ${ds} (see ${log})" | tee -a "${META}"
  fi
}

: > "${META}.jobs"
max_jobs=${#GPUS[@]}
gpu_idx=0

for qlow in "${VALUES[@]}"; do
  for ds in "${DATASETS[@]}"; do
    while (( $(jobs -r -p | wc -l) >= max_jobs )); do
      wait -n || true
    done
    gpu="${GPUS[$gpu_idx]}"
    gpu_idx=$(( (gpu_idx + 1) % max_jobs ))
    run_one "${gpu}" "${qlow}" "${ds}" &
    echo "${qlow} ${ds} ${gpu} $! ${ts}" >> "${META}.jobs"
  done
done
wait || true

echo "=== Summary ===" | tee -a "${META}"
for qlow in "${VALUES[@]}"; do
  tag="${qlow//./p}"
  echo "--- chi2_low=${qlow} ---" | tee -a "${META}"
  sum=0
  cnt=0
  for ds in "${DATASETS[@]}"; do
    log=$(ls scripts/logs/sweep_chi2_low_q${tag}_${ds}_gpu*_${ts}.log 2>/dev/null | head -1)
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
    echo "Average: ${avg} (n=${cnt})" | tee -a "${META}"
  fi
done
echo "done -> ${META}"
