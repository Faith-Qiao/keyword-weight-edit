#!/bin/bash
# queue.sh: run the listed editor runs, at most MAXP concurrent (counting runs already in flight)
cd "$(dirname "$0")"; MAXP=${MAXP:-3}
for spec in "$@"; do
  set -- $spec
  while [ $(pgrep -f "run_wsp.py" | wc -l) -ge $MAXP ]; do sleep 60; done
  [ -d ../out/llm_edit/runs/$1 ] && continue
  ./launch.sh $1 $2 $3; echo "$(date +%H:%M) launched $1"; sleep 30
done
