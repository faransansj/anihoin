"""
characters.json 관리 헬퍼

파일 형식:
{
  "characters": [
    {"key": "folder_name", "tag": "danbooru_search_tag", "display_name": "표시 이름"},
    ...
  ]
}

key   = dataset/raw/<key>/ 폴더명 == 학습 클래스명
tag   = Danbooru 검색 태그
"""

import json
from pathlib import Path

CHARACTERS_FILE = Path("./characters.json")


def load() -> dict[str, dict]:
    """key → {key, tag, display_name} 딕셔너리 반환."""
    if not CHARACTERS_FILE.exists():
        return {}
    with open(CHARACTERS_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return {c["key"]: c for c in data.get("characters", []) if "key" in c}


def save(characters: dict[str, dict]) -> None:
    with open(CHARACTERS_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"characters": list(characters.values())},
            f,
            ensure_ascii=False,
            indent=2,
        )


def get_tags_dict(characters: dict[str, dict]) -> dict[str, str]:
    """크롤러에 전달할 {key: tag} 딕셔너리 반환."""
    return {k: v["tag"] for k, v in characters.items() if v.get("tag")}
