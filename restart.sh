#!/usr/bin/env bash
# Restart the digital-human server.
#
#   ./restart.sh          stop the port owner -> rotate log -> start detached -> probe
#
# Python changes need this; frontend changes do not (GET / re-reads
# static/index.html from disk every request, so a hard-refresh is enough).

set -uo pipefail

PORT="${SERVER_PORT:-17861}"
PYTHON="${PYTHON:-/home/ryu11/.conda/envs/float/bin/python}"
cd "$(dirname "$0")"

# --- 1. stop the PORT OWNER, and wait for it to actually exit -----------------
# By port, not by name: `pgrep -f "...main.py"` also matches the shell running
# the command whenever that pattern is in its own command line, so kill -TERM
# $(pgrep -f ...) can take out your own session.
PID=$(ss -lptnH "sport = :$PORT" | grep -oP 'pid=\K[0-9]+' | head -1)
if [ -n "${PID:-}" ]; then
  kill -TERM "$PID"
  while kill -0 "$PID" 2>/dev/null; do sleep 1; done   # uvicorn drains in ~2s
fi

# --- 2. rotate the log (never truncate with `>`) ------------------------------
# Truncating run.log while the old process still holds it open makes the kernel
# NUL-pad the file up to that process's write offset: the log turns binary and
# plain grep refuses to read it (grep -a still works).
mv -f run.log run.log.1 2>/dev/null

# --- 3. start detached, capturing BOTH stdout and stderr ----------------------
# 2>&1 is not optional: the application log goes to stderr, so a plain
# `python main.py > run.log` captures nothing but the banner.
LD_LIBRARY_PATH=:/usr/local/cuda/lib64 PYTHONUNBUFFERED=1 \
  setsid nohup "$PYTHON" main.py > run.log 2>&1 < /dev/null &

# --- 4. wait for the models (~20s pre-warm), then confirm it answers ----------
# Restarting releases ~4.6 GB of GPU memory and has to take it back; if the box
# is near capacity another job can claim the gap. Watch it come back up.
for _ in $(seq 60); do
  grep -qa "Model pre-warm complete" run.log && break
  sleep 1
done
curl -sk -o /dev/null -w "%{http_code}\n" "https://127.0.0.1:$PORT/"
