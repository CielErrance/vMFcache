#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p scripts/logs

ts=$(date +%Y%m%d_%H%M%S)
LOG="scripts/logs/eta_sweep_ps40_nohup_${ts}.log"
PIDFILE="scripts/logs/eta_sweep_ps40.pid"

nohup bash scripts/run_eta_sweep_ps40.sh > "${LOG}" 2>&1 &
echo $! > "${PIDFILE}"
echo "started pid=$(cat ${PIDFILE})"
echo "log=${LOG}"
