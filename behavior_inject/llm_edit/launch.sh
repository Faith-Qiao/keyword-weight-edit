#!/bin/bash
# usage: launch.sh name budget mode   (detached; log in out/llm_edit/logs/)
cd "$(dirname "$0")"
nohup ../../.venv/bin/python run_wsp.py "$1" "$2" "$3" > ../out/llm_edit/logs/$1.txt 2>&1 &
