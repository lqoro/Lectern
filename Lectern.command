#!/bin/bash
# Double-click to start Lectern and open it.
cd "$(dirname "$0")" || exit 1
PORT="${LECTERN_PORT:-8477}"
if ! curl -s -m 2 "http://127.0.0.1:$PORT/api/status" >/dev/null 2>&1; then
  mkdir -p "$HOME/Library/Application Support/Lectern/logs"
  nohup .venv/bin/python -m lectern \
    >> "$HOME/Library/Application Support/Lectern/logs/server.out" 2>&1 &
  for _ in $(seq 1 60); do
    sleep 0.25
    curl -s -m 2 "http://127.0.0.1:$PORT/api/status" >/dev/null 2>&1 && break
  done
fi
URL="http://127.0.0.1:$PORT/"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
if [ -x "$CHROME" ]; then
  "$CHROME" --app="$URL" >/dev/null 2>&1 &
else
  open "$URL"
fi
