#!/usr/bin/env bash
# Self-orchestrating full-pipeline runner.
#   - waits for any in-progress full null run to finish
#   - runs power --scale full, bugbattery --scale full, then report
# Run once via:  nohup bash run_full.sh > logs/full_orch.log 2>&1 &  (then disown)
set +e
cd "$(dirname "$0")"
export VENV=.venv/bin/python

wait_for_null_full() {
  local need=35
  local seen
  for i in $(seq 1 240); do   # up to 4 h
    if [ -f results/null_calibration.json ]; then
      seen=$("$VENV" - <<'PY'
import json
try:
    d=json.load(open("results/null_calibration.json"))
    print(sum(1 for r in d.get("cells",[]) if not r.get("skipped")))
except Exception:
    print("0")
PY
)
      [ "$seen" = "$need" ] && return 0
    fi
    # also detect a live null process and wait for it to exit
    if pgrep -f "method.py null --scale full" >/dev/null 2>&1; then
      sleep 30; continue
    fi
    sleep 30
  done
  return 1
}

echo "[orch] $(date) waiting for full null run..."
wait_for_null_full
echo "[orch] $(date) null done (rc=$?)"

echo "[orch] $(date) running power --scale full"
"$VENV" method.py power --scale full --workers 4 > logs/power_full.log 2>&1
echo "[orch] $(date) power exit=$?  -> $(grep -c 'done in' logs/power_full.log) cells"

echo "[orch] $(date) running bugbattery --scale full"
"$VENV" method.py bugbattery --scale full --workers 4 > logs/bug_full.log 2>&1
echo "[orch] $(date) bugbattery exit=$?  -> $(grep -c 'done in' logs/bug_full.log) cells"

echo "[orch] $(date) running report"
"$VENV" method.py report > logs/report.log 2>&1
echo "[orch] $(date) report exit=$?"

echo "[orch] $(date) FINISHED"
ls -la results/ 2>/dev/null | tail -20
