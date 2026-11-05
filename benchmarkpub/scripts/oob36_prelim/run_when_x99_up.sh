#!/usr/bin/env bash
# run_when_x99_up.sh — bounded waiter for the step-1.3/1.4 X99-dependent parts.
#
# X99's sshd started refusing the key exchange ("kex_exchange_identification:
# Connection closed by remote host") at ~11:16 CST after a long run of rapid
# connections from this task.  Everything else in step 1.3/1.4 is already done
# and verified locally; only the live remote `sha256sum` comparison and the
# X99 copy of OOB_FOLD_ASSIGNMENT.csv are outstanding.
#
# This script waits (bounded), then runs the remaining pieces.  It never writes
# to X99.

set -uo pipefail

CODE=/home/jingxin/code/dmg-research/project/benchmarkpub/scripts/oob36_prelim
OUT=/home/jingxin/code/dmg-research/project/benchmarkpub/analysis/chapter4_oob36_prelim_20260930
PY=/home/jingxin/code/dmg-research/.venv/bin/python

MAX_WAIT_S=${MAX_WAIT_S:-4200}     # 70 min
INTERVAL_S=${INTERVAL_S:-45}

echo "waiting for X99 ssh (max ${MAX_WAIT_S}s, poll every ${INTERVAL_S}s) ..."
t0=$(date +%s)
ok=0
while :; do
  if timeout 25 ssh -o ConnectTimeout=12 -o BatchMode=yes remote-v100 'echo UP' 2>/dev/null | grep -q UP; then
    ok=1; break
  fi
  now=$(date +%s); el=$(( now - t0 ))
  if [ "$el" -ge "$MAX_WAIT_S" ]; then break; fi
  printf '  %s  still down (%ss elapsed)\n' "$(date +%H:%M:%S)" "$el"
  sleep "$INTERVAL_S"
done

if [ "$ok" -ne 1 ]; then
  echo "X99_STILL_DOWN after ${MAX_WAIT_S}s — step 1.4 not completed"
  exit 2
fi

echo "X99 is back at $(date +%H:%M:%S)"

echo "== re-run 02_download.sh (idempotent) to fetch OOB_FOLD_ASSIGNMENT.csv from X99 =="
bash "$CODE/02_download.sh" || { echo "02_download.sh FAILED"; exit 3; }

echo "== authoritative 03_verify_download.py =="
"$PY" "$CODE/03_verify_download.py"
rc=$?
echo "03_verify_download.py exit=$rc"
exit $rc
