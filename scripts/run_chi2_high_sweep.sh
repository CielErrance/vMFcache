#!/usr/bin/env bash
# chi2_high (empirical delta_high quantile) sweep on FG10 test split.
# Fixed chi2_low=0 (new default). Values: 0.8, 0.9, 0.95, none (--no_delta_high_gate).
# GPUs 4,5,6,7 only.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p scripts/logs

PY=/home/liangyiwen/miniconda3/envs/adapt/bin/python
ts=$(date +%Y%m%d_%H%M%S)
DATA_ROOT=/home/liangyiwen/datasets
GPUS=(4 5 6 7)
# "none" disables the upper gate via --no_delta_high_gate
VALUES=(0.8 0.9 0.95 none)
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
  --chi2_low 0
  --annulus_min_samples 200
  --lambda_div 1.0
  --clip_weight 1.0
)

META="scripts/logs/chi2_high_sweep_${ts}.txt"
echo "chi2_high/delta_high sweep ts=${ts} values=${VALUES[*]} chi2_low=0 fixed GPUs=${GPUS[*]}" | tee "${META}"

run_one() {
  local gpu="$1"
  local qhigh="$2"
  local ds="$3"
  local tag="${qhigh//./p}"
  local log="scripts/logs/sweep_chi2_high_q${tag}_${ds}_gpu${gpu}_${ts}.log"
  local extra=()
  if [[ "${qhigh}" == "none" ]]; then
    extra=(--no_delta_high_gate --chi2_high 0.95)
  else
    extra=(--chi2_high "${qhigh}")
  fi
  echo "[launch] GPU${gpu} chi2_high=${qhigh} ${ds} -> ${log}" | tee -a "${META}"
  if nohup env PYTHONUNBUFFERED=1 WANDB_MODE=disabled "${PY}" ./vMFcache.py \
    "${COMMON[@]}" --gpu "${gpu}" --test_set "${ds}" "${extra[@]}" \
    > "${log}" 2>&1; then
    local acc
    acc=$(grep -E "^${ds}:" "${log}" | tail -1 | awk '{print $2}' || echo "NA")
    echo "[done] GPU${gpu} chi2_high=${qhigh} ${ds} acc=${acc}" | tee -a "${META}"
  else
    echo "[FAIL] GPU${gpu} chi2_high=${qhigh} ${ds} (see ${log})" | tee -a "${META}"
  fi
}

: > "${META}.jobs"
max_jobs=${#GPUS[@]}
gpu_idx=0

for qhigh in "${VALUES[@]}"; do
  for ds in "${DATASETS[@]}"; do
    while (( $(jobs -r -p | wc -l) >= max_jobs )); do
      wait -n || true
    done
    gpu="${GPUS[$gpu_idx]}"
    gpu_idx=$(( (gpu_idx + 1) % max_jobs ))
    run_one "${gpu}" "${qhigh}" "${ds}" &
    echo "${qhigh} ${ds} ${gpu} $! ${ts}" >> "${META}.jobs"
  done
done
wait || true

echo "=== Summary ===" | tee -a "${META}"
for qhigh in "${VALUES[@]}"; do
  tag="${qhigh//./p}"
  echo "--- chi2_high=${qhigh} ---" | tee -a "${META}"
  sum=0
  cnt=0
  for ds in "${DATASETS[@]}"; do
    log=$(ls scripts/logs/sweep_chi2_high_q${tag}_${ds}_gpu*_${ts}.log 2>/dev/null | head -1)
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
