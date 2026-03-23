# HoloScope — Hololive Character Classifier

Swin Transformer-Tiny 기반 홀로라이브 캐릭터 분류기

## 구조

```
anihoin/
├── danbooru_crawler.py   # danbooru SFW 이미지 크롤러
├── dataset.py            # Dataset / Augmentation / DataLoader
├── train.py              # 2-phase fine-tuning 학습 스크립트
├── main.py               # FastAPI 추론 서버
├── demo/
│   └── index.html        # 데모 페이지
└── pyproject.toml
```

## 실행 순서

### 0. 환경 설정
```bash
# uv 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 가상환경 생성 및 의존성 설치 (Python 3.11 자동 사용)
uv sync

# wandb 로깅까지 설치하는 경우
uv sync --extra logging
```

### 1. 크롤링

#### 자격증명 설정 (권장)

익명으로도 실행되지만, Danbooru 계정이 있으면 rate limit이 완화됩니다.

```bash
cp .env.example .env
# .env 파일을 열어 DANBOORU_LOGIN, DANBOORU_API_KEY 입력
# API 키는 https://danbooru.donmai.us/profile 에서 발급
```

#### 실행

```bash
# 기본 실행 (.env 자동 로드)
uv run python danbooru_crawler.py

# 직접 자격증명 전달
uv run python danbooru_crawler.py -u YOUR_NAME -k YOUR_KEY

# 주요 옵션
uv run python danbooru_crawler.py \
  --min-images 300 \
  --max-images 2000 \
  --workers 8 \
  --output-dir ./dataset/raw
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `-u`, `--username` | `""` | Danbooru 사용자명 |
| `-k`, `--api-key` | `""` | Danbooru API 키 |
| `--min-images` | `500` | 미달 시 `others/`로 이동 |
| `--max-images` | `1000` | 캐릭터당 상한 |
| `--workers` | `4` | 병렬 다운로드 수 (최대 권장: 8) |
| `--output-dir` | `./dataset/raw` | 저장 경로 |

**결과 구조:**
```
dataset/raw/
  houshou_marine/   ← min-images 이상 → 분류 대상
  usada_pekora/
  ...
  others/           ← min-images 미만 캐릭터 자동 이동
    himemori_luna/
    ...
```

### 2. 학습
```bash
uv run python train.py \
  --data-dir ./dataset/raw \
  --save-dir ./checkpoints \
  --batch-size 32 \
  --phase1-epochs 5 \
  --phase2-epochs 30 \
  --phase2-lr 1e-5 \
  --patience 7
```

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--phase1-epochs` | `5` | Head-only 학습 epoch 수 |
| `--phase2-epochs` | `30` | Full fine-tune epoch 수 |
| `--phase2-lr` | `1e-5` | Phase 2 학습률 |
| `--patience` | `7` | Early stopping patience (0=비활성화) |
| `--wandb` | `false` | WandB 로깅 활성화 |
| `--wandb-project` | `holoscope` | WandB 프로젝트명 |

**Phase 1** (5 epoch): classification head만 학습
**Phase 2** (30 epoch): 전체 fine-tune, lr=1e-5, early stopping 적용

학습 완료 후 `checkpoints/` 에 저장:
- `best_model.pth` — 최고 val_acc 모델
- `class_map.json` — idx → 캐릭터명 매핑
- `config.json`    — 학습 설정 및 최종 성능

### 3. API 서버 실행
```bash
uv run python main.py
# http://localhost:8000 에서 서버 시작
# http://localhost:8000/docs 에서 Swagger UI 확인
```

### 4. Web UI 데모 실행 (Gradio)
```bash
uv run python demo_gradio.py
# http://localhost:7860 에서 브라우저 접속
# FP32 / FP16 모델 전환 및 API Json을 함께 확인할 수 있습니다.
```

### 5. API 사용 예시

```bash
# 이미지 분류
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@marine.jpg"
```

**Response:**
```json
{
  "file_name": "marine.jpg",
  "confidence": 0.923,
  "meta": {
    "char_name": "Houshou Marine (宝鐘マリン)",
    "cardinal": 19,
    "affiliation": "JP"
  }
}
```

**필드 설명:**

| 필드 | 타입 | 설명 |
|------|------|------|
| `file_name` | string | 업로드된 파일명 |
| `confidence` | float | top-1 예측 신뢰도 (0.0 ~ 1.0) |
| `meta.char_name` | string | 캐릭터 표시명 |
| `meta.cardinal` | int | 해당 지부 내 데뷔 순번 |
| `meta.affiliation` | `JP` \| `EN` \| `IND` \| `null` | 소속 지부 |

**엔드포인트 목록:**

| Method | Path | 설명 |
|--------|------|------|
| `POST` | `/predict` | 이미지 분류 |
| `GET` | `/classes` | 지원 캐릭터 전체 목록 |
| `GET` | `/health` | 서버 상태 및 디바이스 확인 |
| `GET` | `/docs` | Swagger UI |

## 모델 선택 근거

| 항목 | 내용 |
|------|------|
| 모델 | Swin Transformer-Tiny |
| 입력 | 224×224 RGB |
| 사전학습 | ImageNet-1K |
| 파라미터 | ~28M |
| 적합 이유 | ViT 계열 중 데이터 효율 ↑, CNN보다 전역 특징 포착 ↑ |

## 주의사항

- 학술/비상업 목적 데모
- 홀로라이브 캐릭터 © Cover Corp.
- danbooru 크롤링 시 이용약관 준수
