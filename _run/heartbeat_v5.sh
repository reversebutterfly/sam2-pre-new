#!/bin/bash
# v5 paper run heartbeat — emits per-event lines + 8-min idle pulses.
# Configurable via LOGS env var (whitespace-separated list of run.log paths).
#
# Usage:
#   LOGS="vadi_runs/v5_paper_m3/run.log vadi_runs/v5_paper_m2_a3/run.log" \
#     screen -dmS v5-hb bash ~/sam2-pre-new/_run/heartbeat_v5.sh

: ${LOGS:=$(ls ~/sam2-pre-new/vadi_runs/v5_paper_*/run.log 2>/dev/null)}
PATTERN="exported_j_drop|Traceback|done at|joint_search W=|prescreen=|phase_[123]=|local_refine=|OOM|Killed|RuntimeError|nan|inf|python exited rc|infeasible=|\\[a3\\]|collapse_attacked|collapse_control|PRE-REGISTERED VERDICT|d_mem"

# Wait briefly for log files to appear if launching together.
for _ in 1 2 3 4 5; do
  ALL_OK=1
  for log in ${LOGS}; do
    [ -f "${log}" ] || ALL_OK=0
  done
  [ "${ALL_OK}" = "1" ] && break
  sleep 5
done

# Tail each log filtering events.
for log in ${LOGS}; do
  ( tail -F "${log}" 2>/dev/null \
      | grep -E --line-buffered "${PATTERN}" \
      | sed -u "s|^|[$(basename $(dirname ${log}))] |" ) &
done

# 8-min heartbeat with GPU + last-line summary.
( while true; do
    sleep 480
    GPU0_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null)
    GPU1_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 2>/dev/null)
    PY_PROC=$(pgrep -af "run_vadi_v5\\|run_a3_gating" 2>/dev/null \
              | grep -v heartbeat \
              | head -1 \
              | awk "{print \$1}")
    PY_STATE=$([ -n "${PY_PROC}" ] && echo "alive=${PY_PROC}" || echo "DEAD")
    LAST_LINE=""
    for log in ${LOGS}; do
      LL=$(tail -1 "${log}" 2>/dev/null | tr -d "\n" | cut -c-80)
      LAST_LINE="${LAST_LINE} | $(basename $(dirname ${log})): \"${LL}\""
    done
    echo "[hb] $(date +%H:%M:%S) ${PY_STATE} GPU0:${GPU0_MEM}MiB GPU1:${GPU1_MEM}MiB${LAST_LINE}"
done ) &

wait
