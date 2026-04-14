"""크롤링 라우터 — characters.json 기반 (Hololive 비의존)."""

from pathlib import Path

from fastapi import APIRouter, WebSocket

import studio.characters as ch
from studio.jobs.crawl_job import CrawlJob

DATASET_DIR  = Path("./dataset/raw")
ALLOWED_EXT  = {".jpg", ".jpeg", ".png", ".webp"}
PROJECT_ROOT = "."

router = APIRouter(prefix="/crawl", tags=["crawl"])
_job   = CrawlJob()


def _count(key: str) -> int:
    d = DATASET_DIR / key
    if not d.exists():
        return 0
    return sum(1 for f in d.iterdir() if f.suffix.lower() in ALLOWED_EXT)


@router.get("/status")
def get_status():
    return _job.status()


@router.post("/start")
async def start_crawl(body: dict):
    """
    body:
      selected_keys: list[str]   # 크롤할 캐릭터 key 목록 (빈 배열 = 전체)
      min_images: int
      max_images: int
      workers: int
      username: str
      api_key: str
      output_dir: str
    """
    chars = ch.load()

    # 선택된 key만, 없으면 전체
    selected = body.get("selected_keys") or list(chars.keys())
    tags_dict = {k: chars[k]["tag"] for k in selected if k in chars}

    if not tags_dict:
        return {"error": "No characters selected. Add characters first."}

    params = {
        **body,
        "tags_dict": tags_dict,
    }
    await _job.start(params, PROJECT_ROOT)
    return {"started": True, "characters": len(tags_dict)}


@router.post("/stop")
async def stop_crawl():
    await _job.stop()
    return {"stopped": True}


@router.websocket("/logs")
async def crawl_logs(ws: WebSocket):
    await _job.connect_ws(ws)
