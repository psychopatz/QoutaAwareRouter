#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_PORT="7317"
FRONTEND_PORT="5173"
BACKEND_PID=""
FRONTEND_PID=""

kill_port_listeners() {
    local port="$1"
    local pids

    pids="$(ss -ltnp 2>/dev/null | awk -v port=":${port}" '$4 ~ port { print $NF }' | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | sort -u)"

    if [[ -z "$pids" ]]; then
        return 0
    fi

    echo "Stopping existing process(es) on port $port: $pids"
    for pid in $pids; do
        kill "$pid" 2>/dev/null || true
    done

    for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
}

cleanup() {
    local exit_code=$?

    trap - EXIT INT TERM

    if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        kill "$BACKEND_PID" 2>/dev/null || true
    fi

    if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
        kill "$FRONTEND_PID" 2>/dev/null || true
    fi

    wait "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
    exit "$exit_code"
}

trap cleanup EXIT INT TERM

if [[ ! -f "$ROOT_DIR/venv/bin/activate" ]]; then
    echo "Missing Python virtualenv at $ROOT_DIR/venv" >&2
    exit 1
fi

if [[ ! -f "$ROOT_DIR/frontend/package.json" ]]; then
    echo "Missing frontend/package.json" >&2
    exit 1
fi

if [[ ! -d "$ROOT_DIR/frontend/node_modules" ]]; then
    echo "Missing frontend dependencies. Run 'cd frontend && npm install' first." >&2
    exit 1
fi

source "$ROOT_DIR/venv/bin/activate"

kill_port_listeners "$BACKEND_PORT"
kill_port_listeners "$FRONTEND_PORT"

cd "$ROOT_DIR"
PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
python3 -m uvicorn backend.main:app --host 127.0.0.1 --port "$BACKEND_PORT" --reload &
BACKEND_PID=$!

cd "$ROOT_DIR/frontend"
npm run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT" --strictPort &
FRONTEND_PID=$!

echo "Backend started on http://127.0.0.1:$BACKEND_PORT (pid: $BACKEND_PID)"
echo "Frontend started on http://127.0.0.1:$FRONTEND_PORT (pid: $FRONTEND_PID)"
echo "Press Ctrl+C to stop both services."

wait -n "$BACKEND_PID" "$FRONTEND_PID"