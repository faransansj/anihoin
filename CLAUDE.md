# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HoloScope — Swin Transformer-Tiny 기반 홀로라이브 캐릭터 분류기. 학술/비상업 데모 프로젝트.

## Setup

```bash
uv sync                   # 가상환경 생성 + 의존성 설치 (Python 3.11)
uv sync --extra logging   # wandb 포함 설치
```

Optional: Copy `.env.example` to `.env` and set Danbooru credentials (reduces rate limits during crawling).

## Commands

### 1. Crawling
```bash
uv run python danbooru_crawler.py
uv run python danbooru_crawler.py -u USER -k KEY \
  --min-images 500 --max-images 1000 \
  --workers 4 --output-dir ./dataset/raw
```

### 2. Training
```bash
uv run python train.py \
  --data-dir ./dataset/raw \
  --save-dir ./checkpoints \
  --batch-size 32 --phase1-epochs 5 --phase2-epochs 30 --phase2-lr 1e-5
```

### 3. API Server
```bash
uv run python main.py
# → http://localhost:8000  /  http://localhost:8000/docs
```

## Architecture

The pipeline has three stages: **crawl → train → serve**.

### `danbooru_crawler.py`
Fetches SFW images (rating:general,sensitive) from Danbooru for ~60 Hololive members. Characters below `--min-images` are moved to `dataset/raw/others/<char>/`. Auto-loads credentials from `.env` (`DANBOORU_LOGIN`, `DANBOORU_API_KEY`). API pages are fetched sequentially (0.5s sleep), then images are downloaded in parallel via `ThreadPoolExecutor`. Each image is validated with Pillow and deduplicated by MD5 within the session.

### `dataset.py`
- `HoloDataset`: Reads `dataset/raw/` folder structure; folder name = class label. Performs reproducible train/val/test split (80/10/10) by seed. Supports `.jpg`, `.jpeg`, `.png`, `.webp`.
- `build_dataloaders()`: Factory that returns all three DataLoaders. Uses `WeightedRandomSampler` to compensate for class imbalance.
- Augmentation strategy: geometric transforms (flip, rotate, crop) with minimal color-shift (hue ±5°) to preserve character-distinguishing colors.

### `train.py`
Two-phase fine-tuning of `swin_tiny_patch4_window7_224` (ImageNet-1K pretrained, ~28M params):
- **Phase 1** (default 5 epochs): Backbone frozen, only classification head trained (lr=1e-3).
- **Phase 2** (default 30 epochs): Full model unfrozen, low lr (default 1e-5), CosineAnnealingLR.
- Uses mixed-precision (AMP) and CrossEntropyLoss with label_smoothing=0.1.
- Saves `checkpoints/best_model.pth` (best val_acc), `class_map.json` (idx→char), `config.json`.

### `main.py`
FastAPI inference server with:
- `ModelLoader` singleton — loads model + class map once on startup.
- `POST /predict` — accepts image upload, returns top-5 predictions with confidence scores.
- `GET /classes` — lists all supported characters.
- `GET /health` — device info.
- Serves `demo/index.html` at root via `StaticFiles`.

## Path Notes

The README describes a subdirectory layout (`crawler/`, `train/`, `api/`, `demo/`) but all Python files currently reside at the project root. `main.py` has hardcoded relative paths (`../checkpoints`, `../demo`) that assume it runs from an `api/` subdirectory. Adjust these if running directly from root.

## Key Data Files (generated, not committed)

| Path | Description |
|------|-------------|
| `dataset/raw/<char>/` | Training images per character |
| `dataset/raw/others/` | Sub-threshold characters |
| `checkpoints/best_model.pth` | Trained model weights |
| `checkpoints/class_map.json` | `{idx: char_name}` mapping |
| `checkpoints/config.json` | Training config + final metrics |
