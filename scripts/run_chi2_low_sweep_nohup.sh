#!/usr/bin/env bash
# Detached launcher for chi2_low / delta_low quantile sweep.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p scripts/logs

ts=$(date +%Y%m%d_%H%M%S)
LOG="scripts/logs/chi2_low_sweep_nohup_${ts}.log"
PIDFILE="scripts/logs/chi2_low_sweep.pid"

nohup bash scripts/run_chi2_low_sweep.sh > "${LOG}" 2>&1 &
echo $! > "${PIDFILE}"
echo "started pid=$(cat ${PIDFILE})"
echo "log=${LOG}"
echo "meta will be scripts/logs/chi2_low_sweep_<ts>.txt"
echo "tail -f ${LOG}"
