#!/bin/bash
# restart_server.sh <run_name>: wait until that run's model server is idle, kill it, restart with the logged pass count
cd "$(dirname "$0")"; R=$(cd ../out/llm_edit/runs/$1 && pwd); B=$(grep -o '"budget": [0-9]*' $R/meta.json | head -1 | grep -o '[0-9]*')
PID=$(pgrep -f "model_server.py $R ")
while true; do
  c=$(ps -o pcpu= -p $PID | cut -d. -f1); n=$(ls $R/sandbox/rpc/ | grep -c req_)
  if [ "${c:-0}" -lt 5 ] && [ "$n" -eq 0 ]; then sleep 2; c=$(ps -o pcpu= -p $PID | cut -d. -f1); n=$(ls $R/sandbox/rpc/ | grep -c req_); [ "${c:-0}" -lt 5 ] && [ "$n" -eq 0 ] && break; fi
  sleep 1
done
kill $PID; U=$(grep -c '"pass"' $R/server_log.jsonl); rm -f $R/server_ready
nohup ../../.venv/bin/python model_server.py $R $B $U >> $R/server_stdout.txt 2>&1 &
while [ ! -f $R/server_ready ]; do sleep 1; done; echo "$1 restarted at used=$U/$B"
