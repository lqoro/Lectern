#!/bin/bash
# Start Lectern so you can also open it from your iPhone on the same wifi.
# The link printed below contains your access key -- treat it like a password.
cd "$(dirname "$0")" || exit 1
for pid in $(lsof -ti tcp:"${LECTERN_PORT:-8477}" 2>/dev/null); do kill "$pid" 2>/dev/null; done
sleep 1
export LECTERN_HOST=0.0.0.0
exec .venv/bin/python -m lectern
