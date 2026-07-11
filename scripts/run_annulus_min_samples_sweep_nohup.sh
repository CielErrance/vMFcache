#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p scripts/logs

ts=$(date +%Y%m%d_%H%M%S)
LOG="scripts/logs/annulus_min_samples_sweep_nohup_${ts}.log"
PIDFILE="scripts/logs/annulus_min_samples_sweep.pid"

nohup bash scripts/run_annulus_min_samples_sweep.sh > "${LOG}" 2>&1 &
echo $! > "${PIDFILE}"
echo "started pid=$(cat ${PIDFILE})"
echo "log=${LOG}"
