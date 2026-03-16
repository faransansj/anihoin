"""
FastAPI 추론 서버
POST /predict  → 이미지 업로드 → 캐릭터 분류 결과 반환
GET  /classes  → 지원 캐릭터 목록
GET  /health   → 헬스 체크
"""

import io
import json
from pathlib import Path
from typing import Optional

import torch
import timm
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────

CHECKPOINT_DIR = Path("./checkpoints")
MODEL_PATH     = CHECKPOINT_DIR / "best_model.pth"
CLASS_MAP_PATH = CHECKPOINT_DIR / "class_map.json"
CONFIG_PATH    = CHECKPOINT_DIR / "config.json"
IMG_SIZE       = 224
TOP_K          = 5

# 캐릭터 표시 이름 (폴더명 → 출력용)
DISPLAY_NAMES = {
    "tokino_sora":        "Tokino Sora (ときのそら)",
    "roboco":             "Roboco-san (ロボ子さん)",
    "sakura_miko":        "Sakura Miko (さくらみこ)",
    "hoshimachi_suisei":  "Hoshimachi Suisei (星街すいせい)",
    "azki":               "AZKi",
    "yozora_mel":         "Yozora Mel (夜空メル)",
    "shirakami_fubuki":   "Shirakami Fubuki (白上フブキ)",
    "natsuiro_matsuri":   "Natsuiro Matsuri (夏色まつり)",
    "aki_rosenthal":      "Aki Rosenthal (アキ・ローゼンタール)",
    "akai_haato":         "Akai Haato (赤井はあと)",
    "minato_aqua":        "Minato Aqua (湊あくあ)",
    "murasaki_shion":     "Murasaki Shion (紫咲シオン)",
    "nakiri_ayame":       "Nakiri Ayame (百鬼あやめ)",
    "yuzuki_choco":       "Yuzuki Choco (癒月ちょこ)",
    "oozora_subaru":      "Oozora Subaru (大空スバル)",
    "usada_pekora":       "Usada Pekora (兎田ぺこら)",
    "shiranui_flare":     "Shiranui Flare (不知火フレア)",
    "shirogane_noel":     "Shirogane Noel (白銀ノエル)",
    "houshou_marine":     "Houshou Marine (宝鐘マリン)",
    "amane_kanata":       "Amane Kanata (天音かなた)",
    "tsunomaki_watame":   "Tsunomaki Watame (角巻わため)",
    "tokoyami_towa":      "Tokoyami Towa (常闇トワ)",
    "himemori_luna":      "Himemori Luna (姫森ルーナ)",
    "yukihana_lamy":      "Yukihana Lamy (雪花ラミィ)",
    "momosuzu_nene":      "Momosuzu Nene (桃鈴ねね)",
    "shishiro_botan":     "Shishiro Botan (獅白ぼたん)",
    "omaru_polka":        "Omaru Polka (尾丸ポルカ)",
    "laplus_darknesss":   "La+ Darknesss (ラプラス・ダークネス)",
    "takane_lui":         "Takane Lui (鷹嶺ルイ)",
    "hakui_koyori":       "Hakui Koyori (博衣こより)",
    "sakamata_chloe":     "Sakamata Chloe (沙花叉クロヱ)",
    "kazama_iroha":       "Kazama Iroha (風真いろは)",
    "mori_calliope":      "Mori Calliope",
    "takanashi_kiara":    "Takanashi Kiara",
    "ninomae_inanis":     "Ninomae Ina'nis",
    "gawr_gura":          "Gawr Gura",
    "watson_amelia":      "Watson Amelia",
    "ceres_fauna":        "Ceres Fauna",
    "ouro_kronii":        "Ouro Kronii",
    "nanashi_mumei":      "Nanashi Mumei",
    "hakos_baelz":        "Hakos Baelz",
    "others":             "Others (홀로라이브 외)",
}


# ──────────────────────────────────────────────
# 모델 로더 (싱글턴)
# ──────────────────────────────────────────────

class ModelLoader:
    _instance = None

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.idx_to_class: dict[int, str] = {}
        self.transform = None
        self._load()

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self):
        # 클래스 맵 로드
        with open(CLASS_MAP_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        self.idx_to_class = {int(k): v for k, v in raw.items()}
        num_classes = len(self.idx_to_class)

        # 모델 로드
        self.model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            num_classes=num_classes,
        )
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device).eval()

        # 추론용 transform
        self.transform = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        print(f"모델 로드 완료: {num_classes}개 클래스, device={self.device}")

    @torch.no_grad()
    def predict(self, img: Image.Image) -> dict:
        img_np = np.array(img.convert("RGB"))
        tensor = self.transform(image=img_np)["image"].unsqueeze(0).to(self.device)

        with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
            logits = self.model(tensor)

        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        top_k_idx = probs.argsort()[::-1][:TOP_K]

        predicted_idx = int(top_k_idx[0])
        predicted_char = self.idx_to_class[predicted_idx]
        confidence = float(probs[predicted_idx])

        top5 = [
            {
                "character": self.idx_to_class[int(i)],
                "display_name": DISPLAY_NAMES.get(self.idx_to_class[int(i)], self.idx_to_class[int(i)]),
                "confidence": float(probs[i]),
            }
            for i in top_k_idx
        ]

        return {
            "predicted_character": predicted_char,
            "display_name": DISPLAY_NAMES.get(predicted_char, predicted_char),
            "confidence": confidence,
            "is_hololive": predicted_char != "others",
            "top5": top5,
        }


# ──────────────────────────────────────────────
# FastAPI 앱
# ──────────────────────────────────────────────

app = FastAPI(
    title="Hololive Character Classifier",
    description="홀로라이브 캐릭터 분류 API (Swin Transformer-Tiny)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictResponse(BaseModel):
    predicted_character: str
    display_name: str
    confidence: float
    is_hololive: bool
    top5: list[dict]


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    ModelLoader.get()  # 앱 시작 시 모델 로드
    yield


@app.get("/health")
def health():
    return {"status": "ok", "device": str(ModelLoader.get().device)}


@app.get("/classes")
def get_classes():
    loader = ModelLoader.get()
    return {
        "total": len(loader.idx_to_class),
        "classes": [
            {
                "id": idx,
                "character": char,
                "display_name": DISPLAY_NAMES.get(char, char),
            }
            for idx, char in sorted(loader.idx_to_class.items())
        ],
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    # 파일 형식 체크
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 허용됩니다")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="이미지 파일을 읽을 수 없습니다")

    result = ModelLoader.get().predict(img)
    return result


# 데모 페이지 서빙
app.mount("/", StaticFiles(directory="./demo", html=True), name="demo")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
