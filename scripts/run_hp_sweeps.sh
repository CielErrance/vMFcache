#!/usr/bin/env bash
# Run ps_temperature sweep then eta sweep sequentially.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p scripts/logs

ts=$(date +%Y%m%d_%H%M%S)
META="scripts/logs/hp_sweeps_${ts}.txt"
echo "hp sweeps ts=${ts}" | tee "${META}"

echo "[1/2] ps_temperature sweep..." | tee -a "${META}"
bash scripts/run_ps_temperature_sweep.sh 2>&1 | tee -a "${META}"
ps_ts=$(ls -t scripts/logs/ps_temperature_sweep_*.txt 2>/dev/null | head -1 | grep -oP '\d{8}_\d{6}')
echo "[parse] ps_temperature ts=${ps_ts}" | tee -a "${META}"
bash scripts/parse_hp_sweep_results.sh ps_temperature "${ps_ts}" | tee -a "${META}"

echo "[2/2] eta sweep..." | tee -a "${META}"
bash scripts/run_eta_sweep.sh 2>&1 | tee -a "${META}"
eta_ts=$(ls -t scripts/logs/eta_sweep_*.txt 2>/dev/null | head -1 | grep -oP '\d{8}_\d{6}')
echo "[parse] eta ts=${eta_ts}" | tee -a "${META}"
bash scripts/parse_hp_sweep_results.sh eta "${eta_ts}" | tee -a "${META}"

echo "all done -> ${META}"
