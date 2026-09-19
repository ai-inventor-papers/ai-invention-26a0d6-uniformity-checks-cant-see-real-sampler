#!/usr/bin/env bash
# Iteration-2 confirm pipeline: continuation driver (crash-resumable).
#
# Every phase writes its own checkpoint (results/*.json) and records wall_s in
# logs/timings.csv via method.py's _timed wrapper.  The driver skips a phase
# when its marker file exists; markers are created ONLY on exit code 0, so a
# phase that crashed or timed out is re-run on the next invocation.
#
# Launch (detached-safe):
#   setsid nohup bash run_confirm.sh > logs/confirm_orch.log 2>&1 < /dev/null &
set +e
cd "$( dirname "$0" )"
export VENV=.venv/bin/python
export RESERVOIR_CONFIRM_SCALE=full
mkdir -p logs markers
T0=$(date +%s)
note() { echo "[confirm] $(date -u +%H:%M:%S) $*"; }

step() {
  local name=$1; shift
  if [ -f "markers/$name.done" ]; then
    note "$name already done"
    return 0
  fi
  note "START $name"
  "$@" > "logs/c_$name.log" 2>&1
  local rc=$?
  note "$name exit=$rc (tail:)"
  tail -4 "logs/c_$name.log"
  if [ "$rc" = "0" ]; then touch "markers/$name.done"; else note "$name FAILED -- aborting"; exit 1; fi
}

# wait for any in-flight power cell to checkpoint (resume-safe)
wait_power() {
  note "waiting for in-flight power cell to checkpoint..."
  for i in $(seq 1 120); do
    if "$VENV" - <<'EOF' 2>/dev/null
import json, sys
from pathlib import Path
hp = json.loads(Path('results/half_power_law.json').read_text())
cells = {c.get('cell') for c in hp.get('cells', []) if not c.get('skipped')}
want = {'power_cell_m2000_linear_trend','power_cell_m2000_exp_recency',
        'power_cell_m5000_linear_trend','power_cell_m5000_exp_recency'}
sys.exit(0 if want <= cells else 1)
EOF
    then
      note "all m<=5000 power cells present in checkpoint"
      return 0
    fi
    sleep 30
  done
  note "TIMEOUT waiting for power cells -- aborting"
  exit 1
}

wait_power

step power_m10000 "$VENV" method.py confirm power --m 10000 --workers 7 --force
step law        "$VENV" method.py confirm law
step bugs       "$VENV" method.py confirm bugs --workers 7 --force
step pract      "$VENV" method.py confirm pract --cell-b --workers 7 --force
step secondary  "$VENV" method.py confirm secondary --arms 1,2 --workers 7 --force
step gates      "$VENV" method.py confirm gates
step report     "$VENV" method.py confirm report
# NOTE: "python" must NOT be passed as a separate argument -- "$VENV" already
# IS the interpreter; the correct invocation is "$VENV" <script>.
step verify     "$VENV" tools/verify_law.py

T1=$(date +%s)
note "FINISHED in $((T1-T0))s"
ls -la results/ | tail -25