#!/usr/bin/env bash
# ============================================================
# HoloScope 학습 파이프라인 마스터 스크립트
#
# 사용법:
#   bash pipeline/run_pipeline.sh --version v2.0.0
#   bash pipeline/run_pipeline.sh --version v2.0.0 --skip-crawl
#   bash pipeline/run_pipeline.sh --version v2.0.0 --skip-train
#   bash pipeline/run_pipeline.sh --help
# ============================================================

set -euo pipefail

# ── 색상 ────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERR]${RESET}   $*" >&2; exit 1; }
step()    { echo -e "\n${BOLD}━━━ $* ━━━${RESET}"; }

# ── 기본값 ──────────────────────────────────────────────────
VERSION=""
DATA_DIR="./dataset/raw"
CHECKPOINT_DIR="./checkpoints"
SKIP_CRAWL=false
SKIP_TRAIN=false
RELEASE=false
CRAWL_MIN_IMAGES=500
CRAWL_MAX_IMAGES=1000
CRAWL_WORKERS=4
BATCH_SIZE=32
PHASE1_EPOCHS=5
PHASE2_EPOCHS=30
PHASE2_LR=1e-5
PATIENCE=7
XPU_FLAG=""        # Intel Arc XPU 사용 시 --xpu
DEVICE_STR=""      # 디바이스 직접 지정 (xpu:0, cuda:1 등)

usage() {
    cat <<EOF
HoloScope 학습 파이프라인

사용법:
  bash pipeline/run_pipeline.sh [옵션]

필수:
  --version <TAG>         릴리즈 버전 태그 (예: v2.0.0)

크롤링 옵션:
  --skip-crawl            크롤링 단계 건너뛰
  --min-images <N>        캐릭터당 최소 이미지 수 (기본: 500)
  --max-images <N>        캐릭터당 최대 이미지 수 (기본: 1000)
  --crawl-workers <N>     크롤링 병렬 워커 수 (기본: 4)

학습 옵션:
  --skip-train            학습 단계 건너뛰 (이미 학습된 모델 사용)
  --data-dir <DIR>        학습 데이터 경로 (기본: ./dataset/raw)
  --checkpoint-dir <DIR>  체크포인트 저장 경로 (기본: ./checkpoints)
  --batch-size <N>        배치 크기 (기본: 32)
  --phase1-epochs <N>     Phase1 epoch 수 (기본: 5)
  --phase2-epochs <N>     Phase2 epoch 수 (기본: 30)
  --phase2-lr <FLOAT>     Phase2 학습률 (기본: 1e-5)
  --patience <N>          Early stopping patience (기본: 7)

GPU 옵션:
  --xpu                   Intel Arc GPU (XPU) 사용 (IPEX 필요)
  --device <DEV>          디바이스 직접 지정 (예: xpu:0, cuda:1)

릴리스 옵션:
  --release               학습 완료 후 GitHub Release 자동 생성

기타:
  --help                  이 도움말 출력
EOF
    exit 0
}

# ── 인자 파싱 ───────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)       VERSION="$2";           shift 2 ;;
        --data-dir)      DATA_DIR="$2";          shift 2 ;;
        --checkpoint-dir) CHECKPOINT_DIR="$2";   shift 2 ;;
        --skip-crawl)    SKIP_CRAWL=true;        shift ;;
        --skip-train)    SKIP_TRAIN=true;        shift ;;
        --release)       RELEASE=true;           shift ;;
        --min-images)    CRAWL_MIN_IMAGES="$2";  shift 2 ;;
        --max-images)    CRAWL_MAX_IMAGES="$2";  shift 2 ;;
        --crawl-workers) CRAWL_WORKERS="$2";     shift 2 ;;
        --batch-size)    BATCH_SIZE="$2";        shift 2 ;;
        --phase1-epochs) PHASE1_EPOCHS="$2";     shift 2 ;;
        --phase2-epochs) PHASE2_EPOCHS="$2";     shift 2 ;;
        --phase2-lr)     PHASE2_LR="$2";         shift 2 ;;
        --patience)      PATIENCE="$2";          shift 2 ;;
        --xpu)           XPU_FLAG="--xpu";       shift ;;
        --device)        DEVICE_STR="$2";        shift 2 ;;
        --help)          usage ;;
        *) error "알 수 없는 옵션: $1 (--help 참조)" ;;
    esac
done

[[ -z "$VERSION" ]] && error "--version 옵션이 필요합니다. (예: --version v2.0.0)"

# 루트 디렉토리로 이동
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

LOG_FILE="logs/train_${VERSION}_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║         HoloScope 학습 파이프라인                   ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${RESET}"
info "버전:      $VERSION"
info "데이터:    $DATA_DIR"
info "체크포인트: $CHECKPOINT_DIR"
info "로그:      $LOG_FILE"
echo ""

START_TIME=$(date +%s)

# ── STEP 1: 크롤링 ──────────────────────────────────────────
step "STEP 1 / 6  크롤링"
if $SKIP_CRAWL; then
    warn "크롤링 건너뜀 (--skip-crawl)"
else
    if [[ ! -f "crawling/danbooru_crawler.py" ]]; then
        error "crawling/danbooru_crawler.py 없음"
    fi
    info "크롤링 시작 (min=${CRAWL_MIN_IMAGES}, max=${CRAWL_MAX_IMAGES}, workers=${CRAWL_WORKERS})"
    uv run python crawling/danbooru_crawler.py \
        --min-images "$CRAWL_MIN_IMAGES" \
        --max-images "$CRAWL_MAX_IMAGES" \
        --workers    "$CRAWL_WORKERS" \
        --output-dir "$DATA_DIR"
    success "크롤링 완료"
fi

# ── STEP 2: 데이터 통계 ─────────────────────────────────────
step "STEP 2 / 6  데이터 통계"
if [[ ! -d "$DATA_DIR" ]]; then
    error "데이터 디렉토리 없음: $DATA_DIR"
fi

total_classes=0; total_images=0; min_count=9999999; max_count=0
echo ""
printf "  %-35s %8s\n" "캐릭터" "이미지 수"
printf "  %-35s %8s\n" "-------" "--------"
while IFS= read -r -d '' char_dir; do
    char_name=$(basename "$char_dir")
    count=$(find "$char_dir" -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" -o -name "*.webp" \) | wc -l | tr -d ' ')
    printf "  %-35s %8s\n" "$char_name" "$count"
    total_classes=$((total_classes + 1))
    total_images=$((total_images + count))
    (( count < min_count )) && min_count=$count
    (( count > max_count )) && max_count=$count
done < <(find "$DATA_DIR" -maxdepth 1 -mindepth 1 -type d -print0 | sort -z)
echo ""
info "전체 클래스: ${total_classes}  총 이미지: ${total_images}  (최소: ${min_count} / 최대: ${max_count})"

# ── STEP 3: 학습 ────────────────────────────────────────────
step "STEP 3 / 6  학습 (${VERSION})"
if $SKIP_TRAIN; then
    warn "학습 건너뜀 (--skip-train)"
    if [[ ! -f "$CHECKPOINT_DIR/best_model.pth" ]]; then
        error "학습을 건너뛰었지만 $CHECKPOINT_DIR/best_model.pth 가 없습니다."
    fi
else
    info "학습 시작 (로그 → $LOG_FILE)"
    info "모니터링: bash monitor.sh $LOG_FILE"

    # 디바이스 옵션 조합 (XPU_FLAG, DEVICE_STR 다 뱈 가능)
    DEVICE_OPTS=""
    [[ -n "$XPU_FLAG" ]]   && DEVICE_OPTS="$DEVICE_OPTS $XPU_FLAG"
    [[ -n "$DEVICE_STR" ]] && DEVICE_OPTS="$DEVICE_OPTS --device $DEVICE_STR"

    uv run python train.py \
        --data-dir       "$DATA_DIR" \
        --save-dir       "$CHECKPOINT_DIR" \
        --batch-size     "$BATCH_SIZE" \
        --phase1-epochs  "$PHASE1_EPOCHS" \
        --phase2-epochs  "$PHASE2_EPOCHS" \
        --phase2-lr      "$PHASE2_LR" \
        --patience       "$PATIENCE" \
        $DEVICE_OPTS \
        2>&1 | tee "$LOG_FILE"
    success "학습 완료"
fi

# ── STEP 4: FP16 변환 ───────────────────────────────────────
step "STEP 4 / 6  FP16 변환"
uv run python quantize_fp16.py \
    --data-dir "$DATA_DIR" \
    --batch-size "$BATCH_SIZE"
success "FP16 변환 완료"

# ── STEP 5: ONNX 내보내기 ───────────────────────────────────
step "STEP 5 / 6  ONNX 내보내기"
uv run python export_onnx.py \
    --checkpoint-dir "$CHECKPOINT_DIR"
success "ONNX 내보내기 완료"

# ── STEP 6: 결과 요약 ────────────────────────────────────────
step "STEP 6 / 6  결과 요약"
CONFIG="$CHECKPOINT_DIR/config.json"
if [[ -f "$CONFIG" ]]; then
    echo ""
    python3 -c "
import json, sys
with open('$CONFIG') as f:
    c = json.load(f)
print(f'  모델       : swin_tiny_patch4_window7_224')
print(f'  클래스 수  : {c.get(\"num_classes\", \"?\")}')
print(f'  Best val   : {c.get(\"best_val_acc\", 0)*100:.2f}%')
print(f'  Test acc   : {c.get(\"test_acc\", 0)*100:.2f}%')
"
fi

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINUTES=$(( ELAPSED / 60 )); SECONDS=$(( ELAPSED % 60 ))

echo ""
success "파이프라인 완료! (소요: ${MINUTES}분 ${SECONDS}초)"

# ── 자동 릴리즈 ────────────────────────────────────────────
if $RELEASE; then
    step "릴리즈 생성 (${VERSION})"
    bash pipeline/release.sh \
        --version       "$VERSION" \
        --checkpoint-dir "$CHECKPOINT_DIR"
fi

echo ""
info "체크포인트: $CHECKPOINT_DIR/"
info "로그 파일: $LOG_FILE"
