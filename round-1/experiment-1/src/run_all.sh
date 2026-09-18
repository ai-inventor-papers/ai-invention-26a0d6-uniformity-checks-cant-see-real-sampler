#!/usr/bin/env bash
# Final self-contained orchestrator: runs the ENTIRE remaining pipeline in one
# detached session (setsid), fully unattended. Launch ONCE:
#   setsid nohup bash run_all.sh > logs/run_all.log 2>&1 < /dev/null &
set +e
cd "$(dirname "$0")"
export VENV=.venv/bin/python
mkdir -p logs
echo "[run_all] $(date) START null --scale full"
"$VENV" method.py null --scale full --workers 4 > logs/null_full2.log 2>&1
echo "[run_all] $(date) null exit=$? ($(grep -c 'done in' logs/null_full2.log) cells, skipped=$(grep -c 'skipped' logs/null_full2.log))"
echo "[run_all] $(date) START power --scale full"
"$VENV" method.py power --scale full --workers 4 > logs/power_full2.log 2>&1
echo "[run_all] $(date) power exit=$? ($(grep -c 'done in' logs/power_full2.log) cells)"
echo "[run_all] $(date) START bugbattery --scale full"
"$VENV" method.py bugbattery --scale full --workers 4 > logs/bug_full2.log 2>&1
echo "[run_all] $(date) bugbattery exit=$? ($(grep -c 'done in' logs/bug_full2.log) cells)"
echo "[run_all] $(date) START report"
"$VENV" method.py report > logs/report2.log 2>&1
echo "[run_all] $(date) report exit=$?"
echo "[run_all] $(date) FINISHED"
ls -la results/ method_out.json README.md 2>/dev/null
