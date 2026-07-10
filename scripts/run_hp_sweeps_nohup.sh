#!/usr/bin/env bash
# Detached launcher: survives SSH/Cursor disconnect.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p scripts/logs

ts=$(date +%Y%m%d_%H%M%S)
LOG="scripts/logs/hp_sweeps_nohup_${ts}.log"
PIDFILE="scripts/logs/hp_sweeps.pid"

nohup bash scripts/run_hp_sweeps.sh > "${LOG}" 2>&1 &
echo $! > "${PIDFILE}"
echo "started pid=$(cat ${PIDFILE})"
echo "log=${LOG}"
echo "tail -f ${LOG}"
