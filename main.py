"""
FastAPI 추론 서버
POST /predict  → 이미지 업로드 → 캐릭터 분류 결과 반환
GET  /classes  → 지원 캐릭터 목록
GET  /health   → 헬스 체크
"""

import io
import json
from contextlib import asynccontextmanager
from enum import Enum
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
from pydantic import BaseModel

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────

CHECKPOINT_DIR = Path("./checkpoints")
MODEL_PATH     = CHECKPOINT_DIR / "best_model.pth"
CLASS_MAP_PATH = CHECKPOINT_DIR / "class_map.json"
IMG_SIZE       = 224
TOP_K          = 5


# ──────────────────────────────────────────────
# 캐릭터 메타데이터
# cardinal: 해당 지부(branch) 내 데뷔 순번 (1-indexed)
# ──────────────────────────────────────────────

class Affiliation(str, Enum):
    JP  = "JP"
    EN  = "EN"
    IND = "IND"


CHARACTER_META: dict[str, dict] = {
    # ── JP ──────────────────────────────────────
    "tokino_sora":        {"char_name": "Tokino Sora (ときのそら)",               "cardinal": 1,  "affiliation": Affiliation.JP},
    "roboco":             {"char_name": "Roboco-san (ロボ子さん)",                "cardinal": 2,  "affiliation": Affiliation.JP},
    "sakura_miko":        {"char_name": "Sakura Miko (さくらみこ)",               "cardinal": 3,  "affiliation": Affiliation.JP},
    "hoshimachi_suisei":  {"char_name": "Hoshimachi Suisei (星街すいせい)",       "cardinal": 4,  "affiliation": Affiliation.JP},
    "azki":               {"char_name": "AZKi",                                   "cardinal": 5,  "affiliation": Affiliation.JP},
    "yozora_mel":         {"char_name": "Yozora Mel (夜空メル)",                  "cardinal": 6,  "affiliation": Affiliation.JP},
    "shirakami_fubuki":   {"char_name": "Shirakami Fubuki (白上フブキ)",          "cardinal": 7,  "affiliation": Affiliation.JP},
    "natsuiro_matsuri":   {"char_name": "Natsuiro Matsuri (夏色まつり)",          "cardinal": 8,  "affiliation": Affiliation.JP},
    "aki_rosenthal":      {"char_name": "Aki Rosenthal (アキ・ローゼンタール)",   "cardinal": 9,  "affiliation": Affiliation.JP},
    "akai_haato":         {"char_name": "Akai Haato (赤井はあと)",                "cardinal": 10, "affiliation": Affiliation.JP},
    "minato_aqua":        {"char_name": "Minato Aqua (湊あくあ)",                 "cardinal": 11, "affiliation": Affiliation.JP},
    "murasaki_shion":     {"char_name": "Murasaki Shion (紫咲シオン)",            "cardinal": 12, "affiliation": Affiliation.JP},
    "nakiri_ayame":       {"char_name": "Nakiri Ayame (百鬼あやめ)",              "cardinal": 13, "affiliation": Affiliation.JP},
    "yuzuki_choco":       {"char_name": "Yuzuki Choco (癒月ちょこ)",              "cardinal": 14, "affiliation": Affiliation.JP},
    "oozora_subaru":      {"char_name": "Oozora Subaru (大空スバル)",             "cardinal": 15, "affiliation": Affiliation.JP},
    "usada_pekora":       {"char_name": "Usada Pekora (兎田ぺこら)",              "cardinal": 16, "affiliation": Affiliation.JP},
    "shiranui_flare":     {"char_name": "Shiranui Flare (不知火フレア)",          "cardinal": 17, "affiliation": Affiliation.JP},
    "shirogane_noel":     {"char_name": "Shirogane Noel (白銀ノエル)",            "cardinal": 18, "affiliation": Affiliation.JP},
    "houshou_marine":     {"char_name": "Houshou Marine (宝鐘マリン)",            "cardinal": 19, "affiliation": Affiliation.JP},
    "amane_kanata":       {"char_name": "Amane Kanata (天音かなた)",              "cardinal": 20, "affiliation": Affiliation.JP},
    "tsunomaki_watame":   {"char_name": "Tsunomaki Watame (角巻わため)",          "cardinal": 21, "affiliation": Affiliation.JP},
    "tokoyami_towa":      {"char_name": "Tokoyami Towa (常闇トワ)",              "cardinal": 22, "affiliation": Affiliation.JP},
    "himemori_luna":      {"char_name": "Himemori Luna (姫森ルーナ)",             "cardinal": 23, "affiliation": Affiliation.JP},
    "yukihana_lamy":      {"char_name": "Yukihana Lamy (雪花ラミィ)",             "cardinal": 24, "affiliation": Affiliation.JP},
    "momosuzu_nene":      {"char_name": "Momosuzu Nene (桃鈴ねね)",              "cardinal": 25, "affiliation": Affiliation.JP},
    "shishiro_botan":     {"char_name": "Shishiro Botan (獅白ぼたん)",            "cardinal": 26, "affiliation": Affiliation.JP},
    "omaru_polka":        {"char_name": "Omaru Polka (尾丸ポルカ)",               "cardinal": 27, "affiliation": Affiliation.JP},
    "laplus_darknesss":   {"char_name": "La+ Darknesss (ラプラス・ダークネス)",   "cardinal": 28, "affiliation": Affiliation.JP},
    "takane_lui":         {"char_name": "Takane Lui (鷹嶺ルイ)",                  "cardinal": 29, "affiliation": Affiliation.JP},
    "hakui_koyori":       {"char_name": "Hakui Koyori (博衣こより)",              "cardinal": 30, "affiliation": Affiliation.JP},
    "sakamata_chloe":     {"char_name": "Sakamata Chloe (沙花叉クロヱ)",         "cardinal": 31, "affiliation": Affiliation.JP},
    "kazama_iroha":       {"char_name": "Kazama Iroha (風真いろは)",              "cardinal": 32, "affiliation": Affiliation.JP},
    # ── EN ──────────────────────────────────────
    "mori_calliope":      {"char_name": "Mori Calliope",                          "cardinal": 1,  "affiliation": Affiliation.EN},
    "takanashi_kiara":    {"char_name": "Takanashi Kiara",                        "cardinal": 2,  "affiliation": Affiliation.EN},
    "ninomae_inanis":     {"char_name": "Ninomae Ina'nis",                        "cardinal": 3,  "affiliation": Affiliation.EN},
    "gawr_gura":          {"char_name": "Gawr Gura",                              "cardinal": 4,  "affiliation": Affiliation.EN},
    "watson_amelia":      {"char_name": "Watson Amelia",                          "cardinal": 5,  "affiliation": Affiliation.EN},
    "ceres_fauna":        {"char_name": "Ceres Fauna",                            "cardinal": 6,  "affiliation": Affiliation.EN},
    "ouro_kronii":        {"char_name": "Ouro Kronii",                            "cardinal": 7,  "affiliation": Affiliation.EN},
    "nanashi_mumei":      {"char_name": "Nanashi Mumei",                          "cardinal": 8,  "affiliation": Affiliation.EN},
    "hakos_baelz":        {"char_name": "Hakos Baelz",                            "cardinal": 9,  "affiliation": Affiliation.EN},
    # ── others ──────────────────────────────────
    "others":             {"char_name": "Others",                                  "cardinal": 0,  "affiliation": None},
}


# ──────────────────────────────────────────────
# Pydantic 스키마
# ──────────────────────────────────────────────

class CharMeta(BaseModel):
    char_name: str
    cardinal: int
    affiliation: Optional[Affiliation]


class PredictResponse(BaseModel):
    file_name: str
    confidence: float
    meta: CharMeta


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
        with open(CLASS_MAP_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        self.idx_to_class = {int(k): v for k, v in raw.items()}
        num_classes = len(self.idx_to_class)

        self.model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            num_classes=num_classes,
        )
        self.model.load_state_dict(
            torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device).eval()

        self.transform = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

        print(f"모델 로드 완료: {num_classes}개 클래스, device={self.device}")

    @torch.no_grad()
    def predict(self, img: Image.Image) -> tuple[str, float]:
        """(class_key, confidence) 반환"""
        img_np = np.array(img.convert("RGB"))
        tensor = self.transform(image=img_np)["image"].unsqueeze(0).to(self.device)

        with torch.amp.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
            logits = self.model(tensor)

        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        top_idx = int(probs.argmax())
        return self.idx_to_class[top_idx], float(probs[top_idx])


# ──────────────────────────────────────────────
# FastAPI 앱
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app):
    ModelLoader.get()
    yield


app = FastAPI(
    title="Hololive Character Classifier",
    description="홀로라이브 캐릭터 분류 API (Swin Transformer-Tiny)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


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
                "character": char_key,
                **CHARACTER_META.get(char_key, {"char_name": char_key, "cardinal": 0, "affiliation": None}),
            }
            for idx, char_key in sorted(loader.idx_to_class.items())
        ],
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 허용됩니다")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="이미지 파일을 읽을 수 없습니다")

    char_key, confidence = ModelLoader.get().predict(img)
    meta_raw = CHARACTER_META.get(
        char_key,
        {"char_name": char_key, "cardinal": 0, "affiliation": None},
    )

    return PredictResponse(
        file_name=file.filename or "",
        confidence=confidence,
        meta=CharMeta(**meta_raw),
    )


app.mount("/", StaticFiles(directory="./demo", html=True), name="demo")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
