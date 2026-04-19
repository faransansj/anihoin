# 🌟 any-hoin — Integrated Hololive Character Classifier

[한국어](README.kr.md) | [日本語](README.ja.md) | [中文](README.zh.md)

any-hoin is an integrated platform based on Swin Transformer-Tiny for Hololive character classification, allowing the management of data collection (Crawling), model training (Training), and inference services (Inference) through a single integrated web UI.

## 🚀 Quick Start

Follow these commands in order for any OS.

### 1. Environment Setup (Common)
First, install `uv`, the latest Python package manager.

**macOS / Linux**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

**Windows (PowerShell)**
```powershell
powershell -ExecutionPolicy ByPass -Command "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Project Configuration & Dependency Installation
```bash
# Clone repository and move to directory
git clone https://github.com/faransansj/any-hoin.git
cd any-hoin

# Create virtual environment and install dependencies (Python 3.11 automatically set)
uv sync

# (Optional) GPU backend — choose one that matches your hardware
uv sync --extra cuda   # NVIDIA GPU
uv sync --extra rocm   # AMD GPU
uv sync --extra arc    # Intel Arc GPU (XPU)
```

> **Intel Arc users**: After `uv sync --extra arc`, run scripts with `.venv/bin/python` or `uv run --extra arc python` instead of bare `uv run python`. Running without `--extra arc` can cause uv to reinstall the CUDA build of PyTorch and break XPU support.

### 3. Start (One-Shot)

```bash
./start.sh
```

This single command starts both the backend (`http://localhost:8001`) and frontend (`http://localhost:5173`). Press `Ctrl+C` to stop everything at once.

<details>
<summary>Manual startup (if needed)</summary>

**Backend**
```bash
.venv/bin/python studio_api.py
```

**Frontend** (new terminal)
```bash
cd frontend && npm install && npm run dev
```
</details>

---

## 🛠 Integrated Web UI Key Features

You can now control all processes from the web UI without running individual scripts.

- **🌐 Crawl Page**: Selectively collect images for characters from Danbooru.
- **📚 Dataset Page**: Check and manage the status of collected datasets.
- **🏋️ Training Page**: Start model training and monitor training metrics (Loss, Accuracy) in real-time via charts.
- **🔮 Inference Page**: Upload images to classify characters and check metadata instantly.
- **💾 Export Page**: Export trained models and configuration files.

---

## 📂 Project Structure (Latest)

```text
any-hoin/
├── studio/                 # Integrated Backend System
│   ├── jobs/               # Asynchronous Task Processing (Crawling, Training, Export)
│   │   ├── base_job.py     # Base Job Class
│   │   ├── crawl_job.py    # Crawling Job Logic
│   │   └── train_job.py    # Training Job Logic
│   ├── routers/            # API Endpoints (FastAPI)
│   │   ├── characters.py   # Character Metadata Management
│   │   ├── crawl.py        # Crawling Control
│   │   └── training.py     # Training Control
│   └── characters.py       # Character Definitions & Data Models
├── frontend/               # Web UI based on React + TypeScript + Vite
│   ├── src/pages/          # Feature-specific Pages (Crawl, Training, Inference, etc.)
│   └── src/store/          # Job State Management (Zustand)
├── crawling/               # Core Crawling Engine (danbooru_crawler.py)
├── train.py                # Model Training Core Engine
├── dataset.py              # Dataset & Augmentation Pipeline
└── studio_api.py           # Integrated API Server Entry Point
```

---

## ⚙️ Detailed Configuration

### Danbooru Credentials Setup
Setting up the `.env` file is recommended to avoid rate limits.
```bash
cp .env.example .env
# Open .env and enter DANBOORU_LOGIN, DANBOORU_API_KEY
```

## 📊 Model Information
- **Architecture**: Swin Transformer-Tiny
- **Input Size**: 224 $\times$ 224 RGB
- **Pretrained**: ImageNet-1K
- **Key Feature**: Combines high data efficiency of the ViT family with global feature capture capabilities to achieve high classification accuracy even with small amounts of data.

---

## ⚠️ Notice
- This project is a demo for academic and non-commercial purposes.
- All character copyrights belong to © Cover Corp.
- Please comply with the Terms of Service of Danbooru when crawling.
