"""
HoloScope Gradio Web UI
- FP32 / FP16 모델 동적 선택
- 이미지 업로드 시 모델 추론 (Top 1)
- UI용 Label 및 API용 JSON 동시 출력
"""

import json
import time
from pathlib import Path

import gradio as gr
import torch
import timm
import numpy as np
import albumentations as A
from PIL import Image

# ──────────────────────────────────────────────
# 1. 환경 및 메타데이터 설정
# ──────────────────────────────────────────────

CHECKPOINT_DIR = Path("./checkpoints")
MODEL_FP32 = CHECKPOINT_DIR / "best_model.pth"
MODEL_FP16 = CHECKPOINT_DIR / "best_model_fp16.pth"
CLASS_MAP_PATH = CHECKPOINT_DIR / "class_map.json"
IMG_SIZE = 224

# main.py에서 사용하던 메타데이터 가져오기 (단순화를 위해 일부 내장)
# (실제 환경에서는 main.py에서 import 하는 방법도 좋음)
try:
    from main import CHARACTER_META, Affiliation
    HAS_MAIN_META = True
except ImportError:
    HAS_MAIN_META = False
    CHARACTER_META = {}

# ──────────────────────────────────────────────
# 2. 로더 관련
# ──────────────────────────────────────────────

class DemoModelLoader:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.current_precision = None
        self.model = None

        with open(CLASS_MAP_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        self.idx_to_class = {int(k): v for k, v in raw.items()}
        self.num_classes = len(self.idx_to_class)

        self.transform = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def load(self, precision: str):
        if self.current_precision == precision and self.model is not None:
            return  # 이미 로드됨

        print(f"[{precision}] 모델 로드 중... (디바이스: {self.device})")
        
        path = MODEL_FP32 if precision == "FP32" else MODEL_FP16
        if not path.exists():
            raise FileNotFoundError(f"{path} 파일이 없습니다. (quantize_fp16.py 실행 확인)")

        model = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=False,
            num_classes=self.num_classes,
        )
        
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.eval().to(self.device)
        
        # FP16 모델인 경우 파라미터 캐스팅 (CPU는 float16 연산이 제한적이라 예외처리 필요할 수 있음)
        if precision == "FP16" and self.device.type in ["cuda", "mps"]:
            model = model.half()
            
        self.model = model
        self.current_precision = precision
        print(f"[{precision}] 로드 완료")

    def predict(self, img: Image.Image, precision: str) -> tuple[dict, dict]:
        """추론 수행 후 (Label용 dict, JSON용 dict) 반환"""
        self.load(precision)

        img_np = np.array(img.convert("RGB"))
        transformed = self.transform(image=img_np)["image"]          
        inp = transformed.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

        tensor = torch.from_numpy(inp).to(self.device)
        
        if precision == "FP16" and self.device.type in ["cuda", "mps"]:
            tensor = tensor.half()

        t0 = time.time()
        with torch.no_grad():
            logits = self.model(tensor).float() # 안전하게 FP32로 변환 후 softmax
        inf_time = time.time() - t0

        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        # UI 표기용 Dict 생성 (10개)
        top_k = 5
        top_indices = probs.argsort()[-top_k:][::-1]
        
        # UI 레이블 생성용
        labels_dict = {}
        for idx in top_indices:
            char_key = self.idx_to_class[idx]
            if HAS_MAIN_META and char_key in CHARACTER_META:
                display_name = CHARACTER_META[char_key]["char_name"]
            else:
                display_name = char_key
            labels_dict[display_name] = float(probs[idx])

        # 최고 확률 캐릭터
        top_idx = int(top_indices[0])
        top_char_key = self.idx_to_class[top_idx]
        top_conf = float(probs[top_idx])

        # API JSON 포맷
        meta_raw = CHARACTER_META.get(
            top_char_key,
            {"char_name": top_char_key, "cardinal": 0, "affiliation": None},
        )
        
        if isinstance(meta_raw.get("affiliation"), slice) or hasattr(meta_raw.get("affiliation"), "value"):
            meta_raw["affiliation"] = meta_raw["affiliation"].value

        api_json = {
            "file_name": "uploaded_image.jpg",
            "confidence": round(top_conf, 4),
            "meta": meta_raw,
            "_debug": {
                "precision": precision,
                "inference_time_sec": round(inf_time, 4),
                "device": str(self.device)
            }
        }

        return labels_dict, api_json

loader = DemoModelLoader()

# ──────────────────────────────────────────────
# 3. Gradio 인터페이스 구성
# ──────────────────────────────────────────────

def process_image(img, precision):
    if img is None:
        return None, None
    try:
        labels_dict, api_json = loader.predict(img, precision)
        return labels_dict, api_json
    except Exception as e:
        return {"Error": 1.0}, {"error": str(e)}

with gr.Blocks(title="HoloScope Classification UI") as demo:
    gr.Markdown("# 🔍 HoloScope Web Testing UI")
    gr.Markdown("Swin-Transformer Tiny 모델을 이용해 홀로라이브 캐릭터 이미지를 분류합니다.")
    
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="이미지 업로드")
            precision_radio = gr.Radio(
                choices=["FP32", "FP16"], 
                value="FP16", 
                label="추론 모델 포맷 (FP32/FP16)", 
                info="* FP32: best_model.pth / FP16: best_model_fp16.pth"
            )
            submit_btn = gr.Button("분류 (Predict)", variant="primary")
            
        with gr.Column(scale=1):
            ui_output = gr.Label(num_top_classes=5, label="캐릭터 분류 결과 (Top-5)")
            json_output = gr.JSON(label="API 출력 데모 (JSON Response)")

    submit_btn.click(
        fn=process_image,
        inputs=[img_input, precision_radio],
        outputs=[ui_output, json_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
