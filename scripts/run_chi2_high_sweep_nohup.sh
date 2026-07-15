#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p scripts/logs

ts=$(date +%Y%m%d_%H%M%S)
LOG="scripts/logs/chi2_high_sweep_nohup_${ts}.log"
PIDFILE="scripts/logs/chi2_high_sweep.pid"

nohup bash scripts/run_chi2_high_sweep.sh > "${LOG}" 2>&1 &
echo $! > "${PIDFILE}"
echo "started pid=$(cat ${PIDFILE})"
echo "log=${LOG}"
echo "tail -f ${LOG}"
