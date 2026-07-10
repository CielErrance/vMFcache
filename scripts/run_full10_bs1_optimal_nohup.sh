#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p scripts/logs

ts=$(date +%Y%m%d_%H%M%S)
LOG="scripts/logs/full10_bs1_optimal_nohup_${ts}.log"
PIDFILE="scripts/logs/full10_bs1_optimal.pid"

nohup bash scripts/run_full10_bs1_optimal.sh > "${LOG}" 2>&1 &
echo $! > "${PIDFILE}"
echo "started pid=$(cat ${PIDFILE})"
echo "log=${LOG}"
