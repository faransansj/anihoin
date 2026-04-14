# 🌟 HoloScope — Integrated Hololive Character Classifier

HoloScope는 Swin Transformer-Tiny 기반의 홀로라이브 캐릭터 분류기로, 데이터 수집(Crawling), 모델 학습(Training), 그리고 추론 서비스(Inference)를 하나의 통합 웹 UI에서 관리할 수 있는 통합 플랫폼입니다.

## 🚀 Quick Start

어떤 OS에서든 아래 명령어들을 순서대로 복사하여 붙여넣으세요.

### 1. 환경 준비 (공통)
먼저 최신 파이썬 패키지 매니저인 `uv`를 설치합니다.

**macOS / Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 프로젝트 설정 및 의존성 설치
```bash
# 저장소 클론 및 이동
git clone https://github.com/faransansj/anihoin.git
cd anihoin

# 가상환경 생성 및 의존성 설치 (Python 3.11 자동 설정)
uv sync

# (옵션) Intel Arc GPU 사용자의 경우
uv sync --extra arc
```

### 3. 서비스 실행 (All-in-One)

HoloScope는 백엔드 API와 프론트엔드 UI로 구성되어 있습니다.

**Step A: 백엔드 서버 실행**
```bash
uv run python studio_api.py
```
> 서버가 실행되면 `http://localhost:8000`에서 API가 작동합니다.

**Step B: 프론트엔드 UI 실행**
새 터미널을 열고 아래 명령어를 입력하세요.
```bash
cd frontend
npm install
npm run dev
```
> 이제 브라우저에서 `http://localhost:5173` (또는 터미널에 표시된 주소)로 접속하면 통합 관리 화면을 사용할 수 있습니다.

---

## 🛠 통합 웹 UI 주요 기능

이제 개별 스크립트를 실행할 필요 없이 웹 UI에서 모든 과정을 제어할 수 있습니다.

- **🌐 Crawl Page**: 단부루(Danbooru)에서 캐릭터별 이미지를 선택적으로 수집합니다.
- **📚 Dataset Page**: 수집된 데이터셋의 상태를 확인하고 관리합니다.
- **🏋️ Training Page**: 모델 학습을 시작하고, 실시간으로 학습 메트릭(Loss, Accuracy)을 차트로 확인합니다.
- **🔮 Inference Page**: 이미지를 업로드하여 즉시 캐릭터를 분류하고 메타데이터를 확인합니다.
- **💾 Export Page**: 학습된 모델과 설정 파일을 내보냅니다.

---

## 📂 프로젝트 구조 (최신)

```text
anihoin/
├── studio/                 # 통합 백엔드 시스템
│   ├── jobs/               # 비동기 작업 처리 (크롤링, 학습, 내보내기)
│   │   ├── base_job.py     # 작업 기본 클래스
│   │   ├── crawl_job.py    # 크롤링 작업 로직
│   │   └── train_job.py    # 학습 작업 로직
│   ├── routers/            # API 엔드포인트 (FastAPI)
│   │   ├── characters.py   # 캐릭터 메타데이터 관리
│   │   ├── crawl.py        # 크롤링 제어
│   │   └── training.py     # 학습 제어
│   └── characters.py       # 캐릭터 정의 및 데이터 모델
├── frontend/               # React + TypeScript + Vite 기반 웹 UI
│   ├── src/pages/          # 기능별 페이지 (Crawl, Training, Inference 등)
│   └── src/store/          # Job 상태 관리 (Zustand)
├── crawling/               # 핵심 크롤링 엔진 (danbooru_crawler.py)
├── train.py                # 모델 학습 코어 엔진
├── dataset.py              # 데이터셋 및 Augmentation 파이프라인
└── studio_api.py           # 통합 API 서버 진입점
```

---

## ⚙️ 세부 설정

### 단부루 자격증명 설정
크롤링 속도 제한(Rate Limit)을 피하기 위해 `.env` 파일 설정을 권장합니다.
```bash
cp .env.example .env
# .env 파일을 열어 DANBOORU_LOGIN, DANBOORU_API_KEY 입력
```

## 📊 모델 정보
- **Architecture**: Swin Transformer-Tiny
- **Input Size**: 224 $\times$ 224 RGB
- **Pretrained**: ImageNet-1K
- **Key Feature**: ViT 계열의 높은 데이터 효율성과 전역 특징 포착 능력을 결합하여 적은 데이터로도 높은 분류 정확도를 달성합니다.

---

## ⚠️ 주의사항
- 본 프로젝트는 학술 및 비상업적 목적의 데모입니다.
- 모든 캐릭터의 저작권은 © Cover Corp.에 있습니다.
- Danbooru 크롤링 시 해당 사이트의 이용약관을 준수하시기 바랍니다.
