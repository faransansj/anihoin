#!/usr/bin/env bash
# ============================================================
# HoloScope GitHub Release 자동 생성 스크립트
#
# 사전 요구사항:
#   gh CLI 설치 및 로그인 (gh auth login)
#
# 사용법:
#   bash pipeline/release.sh --version v2.0.0
#   bash pipeline/release.sh --version v2.0.0 --dry-run
# ============================================================

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERR]${RESET}   $*" >&2; exit 1; }

# ── 기본값 ──────────────────────────────────────────────────
VERSION=""
CHECKPOINT_DIR="./checkpoints"
DRY_RUN=false

usage() {
    cat <<EOF
HoloScope GitHub Release 자동 생성

사용법:
  bash pipeline/release.sh --version <TAG> [옵션]

필수:
  --version <TAG>         릴리즈 태그 (예: v2.0.0)

옵션:
  --checkpoint-dir <DIR>  체크포인트 위치 (기본: ./checkpoints)
  --dry-run               실제 릴리즈 없이 검증만 수행
  --help                  이 도움말 출력
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)        VERSION="$2";           shift 2 ;;
        --checkpoint-dir) CHECKPOINT_DIR="$2";    shift 2 ;;
        --dry-run)        DRY_RUN=true;           shift ;;
        --help)           usage ;;
        *) error "알 수 없는 옵션: $1" ;;
    esac
done

[[ -z "$VERSION" ]] && error "--version 옵션이 필요합니다."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

echo -e "${BOLD}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║         HoloScope GitHub Release 생성               ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${RESET}"
info "버전: $VERSION"
$DRY_RUN && warn "DRY-RUN 모드 — 실제 릴리즈는 생성되지 않습니다"

# ── 체크: gh CLI 설치 확인 ──────────────────────────────────
if ! command -v gh &>/dev/null; then
    error "gh CLI가 설치되지 않았습니다.\n  설치: brew install gh\n  로그인: gh auth login"
fi

if ! gh auth status &>/dev/null; then
    error "gh CLI 로그인 필요: gh auth login"
fi

# ── 체크: 업로드 파일 존재 확인 ─────────────────────────────
UPLOAD_FILES=(
    "$CHECKPOINT_DIR/best_model.pth"
    "$CHECKPOINT_DIR/best_model_fp16.pth"
    "$CHECKPOINT_DIR/best_model.onnx"
    "$CHECKPOINT_DIR/best_model.onnx.data"
    "$CHECKPOINT_DIR/class_map.json"
    "$CHECKPOINT_DIR/config.json"
)

echo ""
info "업로드 파일 목록:"
missing=false
for f in "${UPLOAD_FILES[@]}"; do
    if [[ -f "$f" ]]; then
        size=$(du -sh "$f" | cut -f1)
        printf "  ${GREEN}✓${RESET} %-45s %s\n" "$f" "$size"
    else
        printf "  ${RED}✗${RESET} %-45s %s\n" "$f" "(없음)"
        missing=true
    fi
done

$missing && error "일부 파일이 없습니다. 파이프라인 전체를 먼저 실행하세요."

# ── 릴리즈 노트 생성 ────────────────────────────────────────
CONFIG="$CHECKPOINT_DIR/config.json"
RELEASE_NOTES=$(python3 - <<PYEOF
import json, datetime

with open("$CONFIG") as f:
    c = json.load(f)

num_classes  = c.get("num_classes", "?")
best_val_acc = c.get("best_val_acc", 0) * 100
test_acc     = c.get("test_acc", 0) * 100
today        = datetime.date.today().isoformat()

print(f"""## HoloScope {VERSION} - Model Weights

릴리즈 일자: {today}

### 성능 지표

| 지표          | 수치             |
|---------------|-----------------|
| 클래스 수     | {num_classes}개  |
| Best Val Acc  | {best_val_acc:.2f}% |
| Test Acc      | {test_acc:.2f}%  |

### 파일 설명

| 파일 | 설명 |
|------|------|
| \`best_model.pth\` | PyTorch FP32 모델 (~105MB) |
| \`best_model_fp16.pth\` | PyTorch FP16 경량화 모델 (~53MB) |
| \`best_model.onnx\` + \`best_model.onnx.data\` | ONNX 모델 |
| \`class_map.json\` | 클래스 인덱스 → 캐릭터 이름 매핑 |
| \`config.json\` | 학습 설정 및 최종 성능 메트릭 |

### 사용법

\`\`\`bash
# checkpoints/ 에 파일 다운로드 후
uv run python main.py
\`\`\`
""")
PYEOF
)

echo ""
info "릴리즈 노트 미리보기:"
echo "───────────────────────────────────────"
echo "$RELEASE_NOTES"
echo "───────────────────────────────────────"

# ── 릴리즈 생성 ─────────────────────────────────────────────
if $DRY_RUN; then
    warn "DRY-RUN: 아래 명령이 실행될 예정입니다."
    echo "  gh release create ${VERSION} \\"
    for f in "${UPLOAD_FILES[@]}"; do
        echo "    \"$f\" \\"
    done
    echo "    --title \"HoloScope ${VERSION}\" \\"
    echo "    --notes '...'"
    success "DRY-RUN 검증 완료"
else
    info "GitHub Release 생성 중..."
    gh release create "$VERSION" \
        "${UPLOAD_FILES[@]}" \
        --title "HoloScope ${VERSION} - Model Weights" \
        --notes "$RELEASE_NOTES"
    success "GitHub Release ${VERSION} 생성 완료!"
    REPO_URL=$(gh repo view --json url -q '.url')
    info "확인: ${REPO_URL}/releases/tag/${VERSION}"
fi
