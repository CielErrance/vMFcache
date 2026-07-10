#!/usr/bin/env bash
# Parse ps_temperature, eta, or rho sweep logs into CSV.
# Usage: bash scripts/parse_hp_sweep_results.sh {ps_temperature|eta|eta_ps40|rho} <timestamp>
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-}"
TS="${2:-}"
if [[ -z "${MODE}" || -z "${TS}" ]]; then
  echo "Usage: $0 {ps_temperature|eta|eta_ps40|rho} <timestamp>"
  exit 1
fi

BASELINE_AVG=70.38
PREFIX="sweep_ps_temp_t"
COL="ps_temperature"
BASELINE_VAL="175"
if [[ "${MODE}" == "eta" ]]; then
  BASELINE_VAL="0.75"
  PREFIX="sweep_eta_e"
  COL="eta"
elif [[ "${MODE}" == "eta_ps40" ]]; then
  BASELINE_AVG=70.54
  BASELINE_VAL="0.75"
  PREFIX="sweep_eta_ps40_e"
  COL="eta"
elif [[ "${MODE}" == "rho" ]]; then
  BASELINE_AVG=70.54
  BASELINE_VAL="2"
  PREFIX="sweep_rho_r"
  COL="rho"
fi

shopt -s nullglob
logs=(scripts/logs/${PREFIX}*_gpu*_${TS}.log)
if [[ ${#logs[@]} -eq 0 ]]; then
  echo "No logs matching scripts/logs/${PREFIX}*_gpu*_${TS}.log"
  exit 1
fi

CSV="scripts/sweep_${MODE}_results_${TS}.csv"
echo "dataset,${COL},accuracy,baseline_avg,delta_vs_baseline" > "${CSV}"

declare -A val_acc
for log in "${logs[@]}"; do
  base=$(basename "${log}")
  if [[ "${MODE}" == "ps_temperature" ]]; then
    val=$(echo "${base}" | sed -n 's/sweep_ps_temp_t\([0-9p]*\)_gpu.*/\1/p' | tr 'p' '.')
  elif [[ "${MODE}" == "rho" ]]; then
    val=$(echo "${base}" | sed -n 's/sweep_rho_r\([0-9p]*\)_gpu.*/\1/p' | tr 'p' '.')
  elif [[ "${MODE}" == "eta_ps40" ]]; then
    val=$(echo "${base}" | sed -n 's/sweep_eta_ps40_e\([0-9p]*\)_gpu.*/\1/p' | tr 'p' '.')
  else
    val=$(echo "${base}" | sed -n 's/sweep_eta_e\([0-9p]*\)_gpu.*/\1/p' | tr 'p' '.')
  fi
  in_sum=0
  avg=""
  while IFS= read -r line; do
    if [[ "${line}" == "=== Evaluation Summary ===" ]]; then
      in_sum=1
      continue
    fi
    if (( in_sum )); then
      if [[ "${line}" == Average:* ]]; then
        avg="${line#Average: }"
      elif [[ "${line}" == *:* ]]; then
        ds="${line%%:*}"
        acc="${line#*: }"
        acc="${acc// /}"
        echo "${ds},${val},${acc},${BASELINE_AVG}," >> "${CSV}"
      fi
    fi
  done < "${log}"
  if [[ -n "${avg}" ]]; then
    val_acc["${val}"]="${avg}"
    echo "Average,${val},${avg},${BASELINE_AVG},$(awk -v a="${avg}" -v b="${BASELINE_AVG}" 'BEGIN{printf "%.2f", a-b}')" >> "${CSV}"
  fi
done

echo "Wrote ${CSV}"
echo "--- Average by ${COL} (baseline ${COL}=${BASELINE_VAL}, ref avg=${BASELINE_AVG}) ---"
for v in $(printf '%s\n' "${!val_acc[@]}" | sort -n); do
  mark=""
  [[ "${v}" == "${BASELINE_VAL}" ]] && mark=" (default)"
  printf "  %s = %s%%%s\n" "${v}" "${val_acc[$v]}" "${mark}"
done
best=$(printf '%s %s\n' "${!val_acc[@]}" "${val_acc[@]}" | awk 'NR%2{val=$1} NR%2{print val,$1}' | sort -k2 -nr | head -1)
echo "Best: ${COL}=$(echo ${best} | awk '{print $1}') avg=$(echo ${best} | awk '{print $2}')%"
