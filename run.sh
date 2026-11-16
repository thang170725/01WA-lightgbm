#!/usr/bin/env bash
# Khởi động toàn bộ hệ thống: backend (FastAPI :8000) + frontend (Next.js :3000)
# Dùng:  ./run.sh            chạy cả hai
#        ./run.sh --train    train lại model trước khi chạy
#        ./run.sh --rebuild  tiền xử lý lại dữ liệu + train lại rồi chạy
set -euo pipefail
cd "$(dirname "$0")"

BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
PY=".venv/bin/python"
MODEL="backend/models/lightgbm_v4.pkl"
LOG_DIR="backend/logs"
mkdir -p "$LOG_DIR"

if [ ! -x "$PY" ]; then
    echo "[run.sh] Chưa có .venv -> tạo mới và cài requirements"
    python3 -m venv .venv
    .venv/bin/pip install -r backend/requirements.txt
fi

if [ ! -d frontend/node_modules ]; then
    echo "[run.sh] Cài node_modules cho frontend"
    (cd frontend && npm install)
fi

# Train nếu được yêu cầu hoặc chưa có model
case "${1:-}" in
    --rebuild) "$PY" -m backend.app.models.lightgbm --rebuild ;;
    --train)   "$PY" -m backend.app.models.lightgbm ;;
    *) [ -f "$MODEL" ] || { echo "[run.sh] Chưa có $MODEL -> train"; "$PY" -m backend.app.models.lightgbm; } ;;
esac

PIDS=()
cleanup() {
    echo; echo "[run.sh] Dừng backend + frontend..."
    for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "[run.sh] Backend  -> http://localhost:${BACKEND_PORT}  (log: $LOG_DIR/backend.log)"
"$PY" -m uvicorn application:app --app-dir backend --host 0.0.0.0 --port "$BACKEND_PORT" >"$LOG_DIR/backend.log" 2>&1 &
PIDS+=($!)

echo "[run.sh] Frontend -> http://localhost:${FRONTEND_PORT}  (log: $LOG_DIR/frontend.log)"
(cd frontend && NEXT_PUBLIC_API_BASE="http://localhost:${BACKEND_PORT}" npm run dev -- -p "$FRONTEND_PORT") >"$LOG_DIR/frontend.log" 2>&1 &
PIDS+=($!)

echo "[run.sh] Nhấn Ctrl+C để dừng tất cả."
wait -n
