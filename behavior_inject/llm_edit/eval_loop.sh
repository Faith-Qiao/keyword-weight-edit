#!/bin/bash
# eval_loop.sh: evaluate each finished WSP run's submitted edit on the test split (one evaluator at a time)
cd "$(dirname "$0")"; O=../out/llm_edit
while true; do
  for r in $O/runs/*; do n=$(basename $r)
    if [ -f $r/submitted_edit.npz ] && grep -q wall_s $r/meta.json 2>/dev/null && [ ! -f $O/results/wsp_$n.json ]; then
      while pgrep -f "evaluate.py" > /dev/null; do sleep 30; done
      ../../.venv/bin/python audit.py $r >> $O/logs/eval_loop.txt 2>&1
      ../../.venv/bin/python evaluate.py edits wsp_$n=$r/submitted_edit.npz 2>&1 | grep -E "on=|Error|Trace" >> $O/logs/eval_loop.txt
      echo "EVALUATED $n" >> $O/logs/eval_loop.txt
    fi
  done
  sleep 60
done
