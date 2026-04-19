#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# .venv/bin/python을 사용해 uv run의 deps 재정렬(CUDA 재설치) 방지
PYTHON=".venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "[error] .venv not found. Run 'uv sync' (or 'uv sync --extra arc/cuda/rocm') first."
    exit 1
fi

# 포트 충돌 시 기존 프로세스 종료
if lsof -ti:8001 &>/dev/null; then
    echo "[info] Killing existing process on port 8001..."
    kill "$(lsof -ti:8001)" 2>/dev/null || true
    sleep 1
fi

# 프론트엔드 의존성 설치 (node_modules 없을 때만)
if [[ ! -d "frontend/node_modules" ]]; then
    echo "[info] Installing frontend dependencies..."
    npm install --prefix frontend
fi

# 백엔드 백그라운드 실행
echo "[info] Starting backend on http://localhost:8001 ..."
"$PYTHON" studio_api.py &
BACKEND_PID=$!

# SIGINT/SIGTERM 시 백엔드도 함께 종료
trap 'echo; echo "[info] Shutting down..."; kill $BACKEND_PID 2>/dev/null; exit 0' INT TERM

# 백엔드 준비 대기 (최대 10초)
for i in $(seq 1 10); do
    if curl -sf http://localhost:8001/api/characters &>/dev/null; then
        break
    fi
    sleep 1
done

# 프론트엔드 포그라운드 실행
echo "[info] Starting frontend on http://localhost:5173 ..."
npm run dev --prefix frontend
