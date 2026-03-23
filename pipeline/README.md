# HoloScope 학습 파이프라인

새로운 버전(v2, v3, ...) 모델을 학습하고 GitHub에 배포하는 전체 파이프라인입니다.

## 흐름

```
[크롤링] → [데이터 통계] → [학습] → [FP16 변환] → [ONNX 내보내기] → [GitHub Release]
```

## 빠른 시작

```bash
# 전체 파이프라인 실행 (크롤링 포함)
bash pipeline/run_pipeline.sh --version v2.0.0

# 이미 데이터가 있는 경우 크롤링 건너뜀
bash pipeline/run_pipeline.sh --version v2.0.0 --skip-crawl

# 학습까지 완료된 경우 내보내기 + 릴리즈만 진행
bash pipeline/run_pipeline.sh --version v2.0.0 --skip-crawl --skip-train

# 완료 후 GitHub Release 자동 생성
bash pipeline/run_pipeline.sh --version v2.0.0 --skip-crawl --release
```

## 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--version` | (필수) | 릴리즈 태그 (예: `v2.0.0`) |
| `--skip-crawl` | false | 크롤링 건너뜀 |
| `--skip-train` | false | 학습 건너뜀 |
| `--release` | false | 완료 후 GitHub Release 자동 생성 |
| `--data-dir` | `./dataset/raw` | 학습 데이터 경로 |
| `--checkpoint-dir` | `./checkpoints` | 체크포인트 저장 경로 |
| `--min-images` | `500` | 캐릭터당 최소 이미지 수 |
| `--max-images` | `1000` | 캐릭터당 최대 이미지 수 |
| `--batch-size` | `32` | 배치 크기 |
| `--phase1-epochs` | `5` | Phase1 epoch 수 |
| `--phase2-epochs` | `30` | Phase2 epoch 수 |
| `--patience` | `7` | Early stopping patience |

## 수동 릴리즈

학습이 완료된 후 릴리즈만 별도 실행하려면:

```bash
# 사전 조건: gh CLI 설치 및 로그인
brew install gh
gh auth login

# 릴리즈 실행 전 검증 (dry-run)
bash pipeline/release.sh --version v2.0.0 --dry-run

# 실제 릴리즈 생성
bash pipeline/release.sh --version v2.0.0
```

## 학습 모니터링

학습 중 별도 터미널에서 실시간 진행 현황을 확인할 수 있습니다:

```bash
# 최신 로그 파일로 모니터링
bash monitor.sh logs/train_v2.0.0_<timestamp>.log
```

## 버전 관리 전략

| 상황 | 권장 버전 표기 |
|------|----------------|
| 새 캐릭터 추가 / 데이터 대폭 증가 | `v2.0.0` |
| 하이퍼파라미터 조정만 | `v1.1.0` |
| 버그 수정 / 경량화만 | `v1.0.1` |
