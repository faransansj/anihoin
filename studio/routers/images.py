"""이미지 업로드 / 이동 / 삭제 / 썸네일 라우터."""

import hashlib
import shutil
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

DATASET_DIR = Path("./dataset/raw")
THUMB_DIR   = Path("./.thumbnails")
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}
THUMB_SIZE  = (200, 200)

router = APIRouter(prefix="/images", tags=["images"])


def _image_files(d: Path) -> list[Path]:
    if not d.exists():
        return []
    return sorted(f for f in d.iterdir() if f.suffix.lower() in ALLOWED_EXT)


# ── 목록 조회 ──────────────────────────────────────────────

@router.get("")
def list_images(
    label: str = Query(...),
    page: int = Query(1, ge=1),
    per_page: int = Query(60, ge=1, le=200),
):
    files = _image_files(DATASET_DIR / label)
    total = len(files)
    start = (page - 1) * per_page
    page_files = files[start : start + per_page]

    return {
        "total":    total,
        "page":     page,
        "per_page": per_page,
        "images": [
            {
                "id":        f"{label}/{f.name}",
                "name":      f.name,
                "label":     label,
                "url":       f"/api/images/file/{label}/{f.name}",
                "thumbnail": f"/api/images/thumb/{label}/{f.name}",
            }
            for f in page_files
        ],
    }


@router.get("/stats")
def get_stats():
    if not DATASET_DIR.exists():
        return {"total": 0, "labels": []}

    labels = []
    total = 0
    for d in sorted(DATASET_DIR.iterdir()):
        if d.is_dir():
            count = len(_image_files(d))
            labels.append({"label": d.name, "count": count})
            total += count

    return {"total": total, "labels": labels}


# ── 업로드 ────────────────────────────────────────────────

@router.post("/upload")
async def upload_images(
    label: str               = Form(...),
    files: list[UploadFile]  = File(...),
):
    label_dir = DATASET_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in ALLOWED_EXT:
            continue
        content  = await f.read()
        md5      = hashlib.md5(content).hexdigest()
        dst_path = label_dir / f"{md5}{ext}"
        if not dst_path.exists():
            dst_path.write_bytes(content)
            saved.append(dst_path.name)

    return {"saved": len(saved), "files": saved}


# ── 이동 / 삭제 ───────────────────────────────────────────

@router.post("/move")
def move_images(body: dict):
    image_ids    = body.get("image_ids", [])
    target_label = body.get("target_label", "")
    if not target_label:
        raise HTTPException(400, "target_label required")

    target_dir = DATASET_DIR / target_label
    target_dir.mkdir(parents=True, exist_ok=True)

    moved = []
    for img_id in image_ids:
        src = DATASET_DIR / img_id
        if src.exists():
            dst = target_dir / src.name
            shutil.move(str(src), str(dst))
            _evict_thumb(img_id)
            moved.append(img_id)

    return {"moved": len(moved)}


@router.delete("")
def delete_images(body: dict):
    image_ids = body.get("image_ids", [])
    deleted = []
    for img_id in image_ids:
        src = DATASET_DIR / img_id
        if src.exists():
            src.unlink()
            _evict_thumb(img_id)
            deleted.append(img_id)
    return {"deleted": len(deleted)}


# ── 파일 / 썸네일 제공 ──────────────────────────────────────

@router.get("/file/{label}/{filename}")
def get_file(label: str, filename: str):
    _guard(label, filename)
    path = DATASET_DIR / label / filename
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(str(path))


@router.get("/thumb/{label}/{filename}")
def get_thumb(label: str, filename: str):
    _guard(label, filename)
    src   = DATASET_DIR / label / filename
    thumb = THUMB_DIR / label / filename

    if not src.exists():
        raise HTTPException(404)

    if not thumb.exists():
        thumb.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(src).convert("RGB")
        img.thumbnail(THUMB_SIZE, Image.LANCZOS)
        img.save(str(thumb), "JPEG", quality=80)

    return FileResponse(str(thumb), media_type="image/jpeg")


# ── 헬퍼 ─────────────────────────────────────────────────

def _guard(label: str, filename: str):
    """Path traversal 방지."""
    for part in (label, filename):
        if ".." in part or "/" in part or "\\" in part:
            raise HTTPException(400, "Invalid path")


def _evict_thumb(img_id: str):
    thumb = THUMB_DIR / img_id
    if thumb.exists():
        thumb.unlink(missing_ok=True)
