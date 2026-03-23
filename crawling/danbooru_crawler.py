"""
Danbooru Crawler for Hololive Character Classification
- SFW only (rating:general,sensitive)
- MD5 중복 제거
- 병렬 이미지 다운로드 (--workers)
- Pillow 이미지 유효성 검사
- .env 자격증명 자동 로드
- Rich CLI UI
"""

import io
import os
import time
import hashlib
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import requests
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import (
    Progress, SpinnerColumn, BarColumn,
    TextColumn, TimeElapsedColumn, MofNCompleteColumn,
)
from rich import box

console = Console()

# ──────────────────────────────────────────────
# 홀로라이브 전체 멤버 태그 목록 (나무위키 문서 존재 기준)
# danbooru 태그 형식: 캐릭터명_(hololive)
# ──────────────────────────────────────────────
HOLOLIVE_MEMBERS = {
    # JP 0기생
    "tokino_sora":        "tokino_sora_(hololive)",
    "roboco":             "roboco-san",
    "sakura_miko":        "sakura_miko",
    "hoshimachi_suisei":  "hoshimachi_suisei",
    "azki":               "azki_(hololive)",

    # JP 1기생
    "yozora_mel":         "yozora_mel",
    "shirakami_fubuki":   "shirakami_fubuki",
    "natsuiro_matsuri":   "natsuiro_matsuri",
    "aki_rosenthal":      "aki_rosenthal",
    "akai_haato":         "akai_haato",

    # JP 2기생
    "minato_aqua":        "minato_aqua",
    "murasaki_shion":     "murasaki_shion",
    "nakiri_ayame":       "nakiri_ayame",
    "yuzuki_choco":       "yuzuki_choco",
    "oozora_subaru":      "oozora_subaru",

    # JP 3기생 (Fantasy)
    "usada_pekora":       "usada_pekora",
    "shiranui_flare":     "shiranui_flare",
    "shirogane_noel":     "shirogane_noel",
    "houshou_marine":     "houshou_marine",

    # JP 4기생
    "amane_kanata":       "amane_kanata",
    "tsunomaki_watame":   "tsunomaki_watame",
    "tokoyami_towa":      "tokoyami_towa",
    "himemori_luna":      "himemori_luna",

    # JP 5기생 (Nepolabo)
    "yukihana_lamy":      "yukihana_lamy",
    "momosuzu_nene":      "momosuzu_nene",
    "shishiro_botan":     "shishiro_botan",
    "omaru_polka":        "omaru_polka",

    # JP 6기생 (Secret Base)
    "laplus_darknesss":   "la+_darknesss",
    "takane_lui":         "takane_lui",
    "hakui_koyori":       "hakui_koyori",
    "sakamata_chloe":     "sakamata_chloe",
    "kazama_iroha":       "kazama_iroha",

    # JP 7기생
    "hiodoshi_ao":        "hiodoshi_ao",
    "otonose_kanade":     "otonose_kanade",
    "ichijou_ririka":     "ichijou_ririka",
    "juufuutei_raden":    "juufuutei_raden",
    "todoroki_hajime":    "todoroki_hajime",

    # EN Myth
    "mori_calliope":      "mori_calliope",
    "takanashi_kiara":    "takanashi_kiara",
    "ninomae_inanis":     "ninomae_ina'nis",
    "gawr_gura":          "gawr_gura",
    "watson_amelia":      "watson_amelia",

    # EN Council / Promise
    "ceres_fauna":        "ceres_fauna",
    "ouro_kronii":        "ouro_kronii",
    "nanashi_mumei":      "nanashi_mumei",
    "hakos_baelz":        "hakos_baelz",

    # EN Advent
    "shiori_novella":     "shiori_novella",
    "koseki_bijou":       "koseki_bijou",
    "nerissa_ravencroft": "nerissa_ravencroft",
    "fuwamoco":           "fuwawa_abyssgard",

    # EN Justice
    "elizabeth_rose_bloodflame": "elizabeth_rose_bloodflame",
    "gigi_murin":         "gigi_murin",
    "cecilia_immergreen": "cecilia_immergreen",
    "raora_panthera":     "raora_panthera",

    # ID Gen1
    "ayunda_risu":        "ayunda_risu",
    "moona_hoshinova":    "moona_hoshinova",
    "airani_iofifteen":   "airani_iofifteen",

    # ID Gen2
    "kureiji_ollie":      "kureiji_ollie",
    "anya_melfissa":      "anya_melfissa",
    "pavolia_reine":      "pavolia_reine",

    # ID Gen3
    "vestia_zeta":        "vestia_zeta",
    "kaela_kovalskia":    "kaela_kovalskia",
    "kobo_kanaeru":       "kobo_kanaeru",

    # DEV_IS ReGLOSS
    "hifumi_ichimatsu":   "ichimatsu_hifumi",
    "isaki_riona":        "isaki_riona",
    "mizumiya_su":        "mizumiya_su",
    "rindo_chihaya":      "rindo_chihaya",
}

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
BASE_URL      = "https://danbooru.donmai.us"
ALLOWED_RATINGS = {"g", "s"}   # general, sensitive (SFW)
ALLOWED_EXT     = {".jpg", ".jpeg", ".png", ".webp"}
PAGE_SLEEP      = 0.5           # API 페이지 요청 간격 (초)


class DanbooruCrawler:
    def __init__(
        self,
        username: str = "",
        api_key: str = "",
        output_dir: Path = Path("./dataset/raw"),
        workers: int = 4,
    ):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "HoloScope-Crawler/1.0"
        self.auth = (username, api_key) if username else None
        self.output_dir = Path(output_dir)
        self.workers = workers
        self._hash_lock = Lock()
        self.seen_hashes: set[str] = set()

    # ── API ──────────────────────────────────

    def _get_posts(self, tag: str, page: int) -> list[dict]:
        params = {
            "tags":  f"{tag} rating:general,sensitive",
            "limit": 100,
            "page":  page,
        }
        try:
            r = self.session.get(
                f"{BASE_URL}/posts.json",
                params=params,
                auth=self.auth,
                timeout=15,
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:
            console.print(f"  [red]API 오류:[/] {e}")
            return []

    # ── 다운로드 ─────────────────────────────

    def _download_one(self, url: str, save_path: Path) -> str:
        """단일 이미지 다운로드. 반환값: 'ok' | 'dup' | 'invalid' | 'error'"""
        try:
            r = self.session.get(url, timeout=20)
            r.raise_for_status()

            # MD5 중복 체크 (세션 내)
            img_hash = hashlib.md5(r.content).hexdigest()
            with self._hash_lock:
                if img_hash in self.seen_hashes:
                    return "dup"
                self.seen_hashes.add(img_hash)

            # Pillow 유효성 검사
            try:
                Image.open(io.BytesIO(r.content)).verify()
            except Exception:
                return "invalid"

            save_path.write_bytes(r.content)
            return "ok"
        except Exception:
            return "error"

    def _collect_queue(
        self, tag: str, char_dir: Path, need: int
    ) -> list[tuple[str, Path]]:
        """다운로드할 (url, 저장경로) 목록을 API에서 수집 (순차적으로 rate-limit 준수)"""
        queue: list[tuple[str, Path]] = []
        page = 1

        while len(queue) < need and page <= 10:
            posts = self._get_posts(tag, page)
            if not posts:
                break

            for post in posts:
                if len(queue) >= need:
                    break

                file_url = post.get("file_url") or post.get("large_file_url")
                if not file_url:
                    continue

                ext = Path(file_url).suffix.lower()
                if ext not in ALLOWED_EXT:
                    continue

                if post.get("rating", "e") not in ALLOWED_RATINGS:
                    continue

                save_path = char_dir / f"{post['id']}{ext}"
                if save_path.exists():
                    continue

                queue.append((file_url, save_path))

            page += 1
            time.sleep(PAGE_SLEEP)

        return queue

    # ── 캐릭터 크롤링 ────────────────────────

    def crawl_character(
        self,
        char_name: str,
        tag: str,
        max_images: int,
        min_images: int,
        progress: Progress,
        task_id,
    ) -> tuple[str, int]:
        """(status, count) 반환.  status: 'included' | 'skipped' | 'below_threshold'"""
        char_dir = self.output_dir / char_name
        char_dir.mkdir(parents=True, exist_ok=True)

        existing = len([
            f for f in char_dir.iterdir()
            if f.suffix.lower() in ALLOWED_EXT
        ])

        # 이미 충분히 수집된 경우
        if existing >= max_images:
            progress.update(
                task_id,
                completed=max_images,
                total=max_images,
                description=f"[dim]{char_name:<26}[/]",
            )
            return "skipped", existing

        collected = existing
        progress.update(task_id, completed=collected, total=max_images)

        # 필요한 이미지 목록 수집 (API 순차 호출)
        queue = self._collect_queue(tag, char_dir, max_images - collected)

        # 병렬 다운로드
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(self._download_one, url, path): path
                for url, path in queue
            }
            for future in as_completed(futures):
                if future.result() == "ok":
                    collected += 1
                    progress.update(task_id, completed=min(collected, max_images))

        if collected < min_images:
            return "below_threshold", collected
        return "included", collected

    # ── 전체 실행 ────────────────────────────

    def run(self, min_images: int = 500, max_images: int = 1000) -> dict:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        auth_label = (
            f"[cyan]{self.auth[0]}[/]" if self.auth else "[yellow]익명[/yellow] (rate limit 있음)"
        )
        console.print(
            Panel.fit(
                f"[bold white]HoloScope Danbooru Crawler[/]\n"
                f"계정 [dim]│[/] {auth_label}\n"
                f"대상 [dim]│[/] [cyan]{len(HOLOLIVE_MEMBERS)}명[/]  "
                f"기준 [dim]│[/] [green]{min_images}[/]–[green]{max_images}[/]장  "
                f"워커 [dim]│[/] [cyan]{self.workers}[/]개",
                border_style="cyan",
                title="[bold cyan]설정[/]",
            )
        )
        console.print()

        results: dict[str, list] = {
            "included":        [],
            "skipped":         [],
            "below_threshold": [],
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=26),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            overall = progress.add_task(
                f"[bold cyan]{'전체 진행':<26}[/]",
                total=len(HOLOLIVE_MEMBERS),
            )

            for char_name, tag in HOLOLIVE_MEMBERS.items():
                task_id = progress.add_task(
                    f"[yellow]{char_name:<26}[/]",
                    total=max_images,
                )

                status, count = self.crawl_character(
                    char_name, tag, max_images, min_images, progress, task_id
                )

                if status == "included":
                    progress.update(
                        task_id,
                        description=f"[green]✓ {char_name:<25}[/]",
                    )
                    results["included"].append((char_name, count))

                elif status == "skipped":
                    progress.update(
                        task_id,
                        description=f"[dim]  {char_name:<25}[/]",
                    )
                    results["skipped"].append((char_name, count))

                else:  # below_threshold
                    progress.update(
                        task_id,
                        description=f"[red]✗ {char_name:<25}[/]",
                    )
                    src = self.output_dir / char_name
                    dst = self.output_dir / "others" / char_name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if src.exists():
                        shutil.move(str(src), str(dst))
                    results["below_threshold"].append((char_name, count))

                progress.update(overall, advance=1)

        self._print_summary(results)
        return results

    def _print_summary(self, results: dict) -> None:
        console.print()
        console.rule("[bold]크롤링 결과[/]", style="cyan")
        console.print()

        if results["included"]:
            t = Table(
                title=f"포함된 캐릭터 ({len(results['included'])}명)",
                box=box.ROUNDED,
                border_style="green",
                header_style="bold green",
                show_lines=False,
            )
            t.add_column("캐릭터", style="white", min_width=26)
            t.add_column("이미지", justify="right", style="bold green")
            for name, cnt in sorted(results["included"], key=lambda x: -x[1]):
                t.add_row(name, f"{cnt:,}")
            console.print(t)

        if results["below_threshold"]:
            console.print()
            t2 = Table(
                title=f"기준 미달 → others/ ({len(results['below_threshold'])}명)",
                box=box.ROUNDED,
                border_style="yellow",
                header_style="bold yellow",
            )
            t2.add_column("캐릭터", style="dim", min_width=26)
            t2.add_column("이미지", justify="right", style="yellow")
            for name, cnt in results["below_threshold"]:
                t2.add_row(name, f"{cnt:,}")
            console.print(t2)

        if results["skipped"]:
            console.print(
                f"[dim]이미 완료된 캐릭터: {len(results['skipped'])}명 (스킵됨)[/]"
            )
        console.print()


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="홀로라이브 캐릭터 이미지 크롤러 (Danbooru SFW)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python danbooru_crawler.py                         # 익명 실행 (rate limit 있음)
  python danbooru_crawler.py -u USER -k KEY          # 인증 실행 (권장)
  python danbooru_crawler.py --min-images 300        # 최소 기준 완화
  python danbooru_crawler.py --max-images 2000       # 더 많이 수집
  python danbooru_crawler.py --workers 8             # 병렬 다운로드 증가

자격증명은 .env 파일에 미리 설정해도 됩니다:
  DANBOORU_LOGIN=your_username
  DANBOORU_API_KEY=your_api_key
        """,
    )
    parser.add_argument(
        "-u", "--username",
        default=os.getenv("DANBOORU_LOGIN", ""),
        metavar="USER",
        help="Danbooru 사용자명 (또는 .env DANBOORU_LOGIN)",
    )
    parser.add_argument(
        "-k", "--api-key",
        default=os.getenv("DANBOORU_API_KEY", ""),
        metavar="KEY",
        help="Danbooru API 키 (또는 .env DANBOORU_API_KEY)",
    )
    parser.add_argument(
        "--min-images",
        type=int, default=500, metavar="N",
        help="캐릭터 포함 최소 이미지 수 (기본: 500)",
    )
    parser.add_argument(
        "--max-images",
        type=int, default=1000, metavar="N",
        help="캐릭터당 최대 수집 이미지 수 (기본: 1000)",
    )
    parser.add_argument(
        "--output-dir",
        default="./dataset/raw", metavar="DIR",
        help="이미지 저장 경로 (기본: ./dataset/raw)",
    )
    parser.add_argument(
        "--workers",
        type=int, default=4, metavar="N",
        help="병렬 다운로드 스레드 수 (기본: 4, 최대 권장: 8)",
    )
    args = parser.parse_args()

    DanbooruCrawler(
        username=args.username,
        api_key=args.api_key,
        output_dir=Path(args.output_dir),
        workers=args.workers,
    ).run(min_images=args.min_images, max_images=args.max_images)
