"""모델 양자화 / ONNX export 라우터."""

import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, WebSocket
from fastapi.responses import FileResponse

from studio.jobs.export_job import Fp16Job, OnnxJob

CHECKPOINT_DIR = Path("./checkpoints")
PROJECT_ROOT   = "."

router    = APIRouter(prefix="/export", tags=["export"])
_fp16_job = Fp16Job()
_onnx_job = OnnxJob()


# ── 모델 목록 ─────────────────────────────────────────────

@router.get("/models")
def list_models():
    entries = {
        "fp32": "best_model.pth",
        "fp16": "best_model_fp16.pth",
        "onnx": "best_model.onnx",
    }
    result = {}
    for key, fname in entries.items():
        p = CHECKPOINT_DIR / fname
        result[key] = {
            "exists":   p.exists(),
            "size_mb":  round(p.stat().st_size / 1024**2, 1) if p.exists() else None,
            "filename": fname,
        }
    return {"models": result}


# ── 변환 실행 ─────────────────────────────────────────────

@router.post("/fp16")
async def export_fp16():
    await _fp16_job.start(PROJECT_ROOT)
    return {"started": True}


@router.post("/onnx")
async def export_onnx(body: dict = {}):
    opset = int(body.get("opset", 18))
    await _onnx_job.start(opset, PROJECT_ROOT)
    return {"started": True}


@router.post("/fp16/stop")
async def stop_fp16():
    await _fp16_job.stop()
    return {"stopped": True}


@router.post("/onnx/stop")
async def stop_onnx():
    await _onnx_job.stop()
    return {"stopped": True}


# ── 상태 ─────────────────────────────────────────────────

@router.get("/status")
def get_status():
    return {"fp16": _fp16_job.status(), "onnx": _onnx_job.status()}


# ── 다운로드 ─────────────────────────────────────────────

@router.get("/download/{filename}")
def download_model(filename: str):
    if ".." in filename or "/" in filename:
        raise HTTPException(400, "Invalid filename")
    path = CHECKPOINT_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Model not found")
    return FileResponse(str(path), filename=filename, media_type="application/octet-stream")


# ── WebSocket 로그 ────────────────────────────────────────

@router.websocket("/logs/fp16")
async def fp16_logs(ws: WebSocket):
    await _fp16_job.connect_ws(ws)


@router.websocket("/logs/onnx")
async def onnx_logs(ws: WebSocket):
    await _onnx_job.connect_ws(ws)
