#!/usr/bin/env bash

# Khởi động hệ thống:
# Backend  : FastAPI :8000
# Frontend : Next.js :3000
#
# Sử dụng:
#   ./run.sh

set -Eeuo pipefail

cd "$(dirname "$0")"

# ==============================
# CONFIG
# ==============================

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

PY=".venv/bin/python"

MODEL_BASE="/home/thang/workspace/lightgbm/v0_0_1/result/r0_0_1_1"
MODELS=(
    "$MODEL_BASE/r_low/models/lightgbm_low.pkl"
    "$MODEL_BASE/r_medium/models/lightgbm_medium.pkl"
    "$MODEL_BASE/r_high/models/lightgbm_high.pkl"
)

LOG_DIR="backend/logs"

mkdir -p "$LOG_DIR"

# ==============================
# CHECK ENVIRONMENT
# ==============================

if [ ! -x "$PY" ]; then
    echo "[ERROR] Không tìm thấy Python virtual environment: $PY"
    exit 1
fi

if [ ! -d "frontend/node_modules" ]; then
    echo "[ERROR] Chưa cài frontend/node_modules"
    echo "Chạy: cd frontend && npm install"
    exit 1
fi

for m in "${MODELS[@]}"; do
    if [ ! -f "$m" ]; then
        echo "[ERROR] Không tìm thấy model:"
        echo "$m"
        exit 1
    fi
done

# ==============================
# PROCESS MANAGEMENT
# ==============================

PIDS=()

cleanup() {
    echo
    echo "[run.sh] Đang dừng hệ thống..."

    for pid in "${PIDS[@]:-}"; do
        pkill -P "$pid" 2>/dev/null || true
        kill "$pid" 2>/dev/null || true
    done

    wait 2>/dev/null || true

    echo "[run.sh] Đã dừng hệ thống."
}

trap cleanup INT TERM EXIT

# ==============================
# START BACKEND
# ==============================

echo "[run.sh] Starting FastAPI..."

"$PY" -m uvicorn application:app \
    --app-dir backend \
    --host 127.0.0.1 \
    --port "$BACKEND_PORT" \
    --workers 1 \
    > "$LOG_DIR/backend.log" 2>&1 &

PIDS+=("$!")

# ==============================
# START FRONTEND
# ==============================

echo "[run.sh] Starting Next.js..."

rm -f frontend/.next/dev/lock

NEXT_PUBLIC_API_BASE="http://localhost:${BACKEND_PORT}" \
    npm --prefix frontend run dev -- -p "$FRONTEND_PORT" \
    > "$LOG_DIR/frontend.log" 2>&1 &

PIDS+=("$!")

# ==============================
# STATUS
# ==============================

echo
echo "======================================"
echo " SYSTEM STARTED"
echo "======================================"
echo "Backend  : http://localhost:${BACKEND_PORT}"
echo "Frontend : http://localhost:${FRONTEND_PORT}"
echo "Models   : $MODEL_BASE (low, medium, high)"
echo "Backend log  : $LOG_DIR/backend.log"
echo "Frontend log : $LOG_DIR/frontend.log"
echo "======================================"
echo
echo "Nhấn Ctrl+C để dừng hệ thống."

wait -n "${PIDS[@]}"
