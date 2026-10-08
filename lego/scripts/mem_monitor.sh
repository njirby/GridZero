#!/usr/bin/env bash
# Every 15s: RAM/swap totals, GPU mem, top-8 RSS procs.  Usage: bash mem_monitor.sh [logfile]
LOG="${1:-/home/nate/Documents/GridZero/lego/logs/mem.log}"; mkdir -p "$(dirname "$LOG")"
while true; do
  { echo "=== $(date +%T) $(free -m | awk '/Mem:/{printf "ram_used=%dM avail=%dM",$3,$7} /Swap:/{printf " swap_used=%dM",$3}')"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' '; echo
    ps -eo rss=,comm=,args= --sort=-rss | head -8 | awk '{printf "  %6dM %s\n",$1/1024,substr($0,index($0,$2),90)}'
  } >> "$LOG"; sleep 15
done
