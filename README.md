# AniHoin — Hololive Image Crawler

Danbooru에서 홀로라이브 캐릭터 이미지를 수집하는 CLI 툴입니다.  
YOLO 학습을 위한 데이터셋 구성에 사용됩니다.

---

## 구조

```
anihoin/
├── crawler/
│   ├── __init__.py
│   ├── hololive_tags.py      # 캐릭터 태그 정의 (JP/EN/ID 전 세대)
│   ├── danbooru_client.py    # Danbooru API 클라이언트
│   └── cli.py                # CLI 인터페이스
├── main.py                   # 엔트리포인트
├── requirements.txt
├── .env.example              # API 키 설정 예시
└── data/                     # 다운로드된 이미지 저장 위치
    ├── tokino_sora/
    ├── usada_pekora/
    └── ...
```

---

## 설치

### 요구사항

- Python 3.10+

### 의존성 설치

```bash
pip install -r requirements.txt
```

---

## Danbooru API 키 설정 (선택, 권장)

인증 없이도 사용할 수 있지만, 익명 사용 시 다음 제한이 있습니다:

| 항목 | 익명 | 인증 (Gold+) |
|------|------|------------|
| 요청 태그 수 | 최대 2개 | 최대 6개 |
| 페이지당 이미지 | 20개 | 200개 |
| 요청 딜레이 | 2초 | 1초 |

### API 키 발급

1. [Danbooru](https://danbooru.donmai.us) 에 로그인
2. 우측 상단 프로필 → **My Account** → **API Key** 섹션에서 생성
3. 프로젝트 루트에 `.env` 파일 생성:

```bash
cp .env.example .env
```

`.env` 파일을 열어 본인 정보로 수정:

```ini
DANBOORU_LOGIN=your_username
DANBOORU_API_KEY=your_api_key
```

---

## 사용법

### 기본 실행 — 인터랙티브 모드

```bash
python main.py
```

1. Danbooru에서 전 캐릭터 이미지 수를 조회합니다
2. Rich 테이블로 캐릭터별 이미지 수를 확인합니다
3. 체크박스로 원하는 캐릭터를 선택합니다
4. 각 캐릭터당 다운로드 수량을 입력합니다
5. 확인 후 다운로드를 진행합니다

---

### CLI 옵션

| 옵션 | 단축키 | 기본값 | 설명 |
|------|--------|--------|------|
| `--list` | `-l` | — | 이미지 수만 조회하고 종료 |
| `--download TAG` | `-d TAG` | — | 특정 캐릭터 태그 다운로드 (반복 사용 가능) |
| `--all` | `-a` | — | 활동 중인 모든 캐릭터 다운로드 |
| `--limit N` | `-n N` | `200` | 캐릭터당 최대 이미지 수 |
| `--rating RATING` | `-r RATING` | `general` | 등급 필터 (`general` / `safe` / `questionable` / `any`) |
| `--output DIR` | `-o DIR` | `./data` | 저장 디렉토리 |
| `--no-cache` | — | — | 캐시 무시, 이미지 수 재조회 |
| `--include-retired` | — | — | 졸업/은퇴 멤버 포함 |
| `--min-count N` | — | `0` | N개 미만 캐릭터는 목록에서 숨김 |
| `--verbose` | `-v` | — | 상세 디버그 로그 출력 |

---

### 사용 예시

```bash
# 캐릭터 목록 및 이미지 수 확인
python main.py --list

# 이미지가 1000개 이상인 캐릭터만 표시
python main.py --list --min-count 1000

# 특정 캐릭터 2명, 각 300장 다운로드
python main.py --download tokino_sora --download usada_pekora --limit 300

# 전체 캐릭터 200장씩, safe 등급 이미지 다운로드
python main.py --all --limit 200 --rating safe

# 저장 경로 지정
python main.py --all --limit 500 --output ./dataset/hololive

# 졸업 멤버 포함, 캐시 재조회
python main.py --list --include-retired --no-cache
```

---

## 출력 디렉토리 구조

다운로드된 이미지는 **캐릭터 태그** 이름의 폴더에 저장됩니다.  
파일명은 Danbooru 포스트 ID를 사용합니다.

```
data/
├── tokino_sora/
│   ├── 1234567.jpg
│   ├── 2345678.png
│   └── ...
├── usada_pekora/
│   ├── 3456789.jpg
│   └── ...
└── others/          # (직접 추가, YOLO 학습용 기타 클래스)
```

### 이미지 수 캐시

처음 실행 시 Danbooru에서 이미지 수를 조회한 결과가 `character_counts.json`에 캐시됩니다.  
이후 실행에서는 캐시를 사용해 빠르게 목록을 표시합니다.  
최신 수치를 보려면 `--no-cache` 플래그를 사용하세요.

---

## 지원 캐릭터

현재 지원하는 홀로라이브 멤버:

| 브랜치 | 세대 |
|--------|------|
| **JP** | Gen 0, Gen 1, Gen 2, GAMERS, Gen 3, Gen 4, Gen 5, Gen 6 |
| **DEV_IS** | ReGLOSS, FLOW GLOW |
| **EN** | Myth, Project: HOPE, Council, Advent, Justice |
| **ID** | Gen 1, Gen 2, Gen 3 |

졸업/은퇴 멤버는 기본적으로 제외됩니다. `--include-retired`로 포함할 수 있습니다.

---

## YOLO 데이터셋 구성 예시

수집된 이미지를 YOLO 학습 데이터로 변환하는 기본 흐름:

```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

`data.yaml` 예시:

```yaml
path: ./dataset
train: images/train
val: images/val

nc: 70  # 캐릭터 수 + others
names:
  - tokino_sora
  - usada_pekora
  # ... 기타 캐릭터 태그 순서대로
  - others
```

> **Note:** 이미지 수집 후 바운딩박스 어노테이션이 필요합니다.  
> Danbooru에서 받은 이미지는 캐릭터가 중심에 있는 일러스트가 많아 자동 crop이 가능합니다.

---

## 주의사항

- Danbooru의 이용 약관을 준수하세요. 과도한 요청은 IP 차단 원인이 됩니다.
- 수집된 이미지의 저작권은 각 작가에게 있습니다. **모델 학습 목적**으로만 사용하세요.
- 기본 등급 필터는 `general`(전체이용가)입니다. `questionable` 이상은 명시적으로 설정해야 합니다.

---

## 라이선스

MIT License