#!/usr/bin/env bash
# Start the FastAPI backend and Vite frontend dev server concurrently.
set -e

ROOT=$(cd "$(dirname "$0")" && pwd)
BACKEND_PORT=8000
FRONTEND_PORT=5173

cleanup_port() {
  local port="$1"
  local pids
  pids=$(lsof -ti tcp:"$port" || true)

  if [ -n "$pids" ]; then
    echo "Port $port is already in use, stopping process(es): $pids"
    kill $pids || true
  fi
}

cleanup() {
  kill "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
}

echo "Installing Python dependencies ..."
cd "$ROOT"
pip install -r requirements.txt

echo "Installing WebUI dependencies ..."
cd "$ROOT/webui" && npm install

cleanup_port "$BACKEND_PORT"
cleanup_port "$FRONTEND_PORT"

cd "$ROOT"
echo "Starting FastAPI backend on http://localhost:$BACKEND_PORT ..."
uvicorn api.main:app --host 0.0.0.0 --port "$BACKEND_PORT" --reload &
BACKEND_PID=$!

echo "Starting Vite frontend on http://localhost:$FRONTEND_PORT ..."
cd "$ROOT/webui" && npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT" &
FRONTEND_PID=$!

trap cleanup INT TERM EXIT
wait "$BACKEND_PID" "$FRONTEND_PID"
