"""CrawlJob — danbooru_crawler.py subprocess 래퍼 (범용 캐릭터 지원)."""

import asyncio
import json
import os
import tempfile

from .base_job import BaseJob


class CrawlJob(BaseJob):
    def __init__(self):
        super().__init__("crawl")
        self._tmp_file: str | None = None

    async def start(self, params: dict, project_root: str):
        if self.state == "running":
            return
        self.state = "running"

        # 이전 임시 파일 정리
        self._cleanup_tmp()

        cmd = ["uv", "run", "python", "crawling/danbooru_crawler.py"]

        if params.get("username"):
            cmd += ["-u", params["username"]]
        if params.get("api_key"):
            cmd += ["-k", params["api_key"]]

        cmd += ["--min-images", str(params.get("min_images", 500))]
        cmd += ["--max-images", str(params.get("max_images", 1000))]
        cmd += ["--workers",    str(params.get("workers", 4))]
        cmd += ["--output-dir", params.get("output_dir", "./dataset/raw")]

        # {key: tag} 딕셔너리를 임시 JSON 파일로 전달
        tags_dict: dict[str, str] = params.get("tags_dict", {})
        if tags_dict:
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False, prefix="studio_tags_"
            )
            json.dump(tags_dict, tmp, ensure_ascii=False)
            tmp.close()
            self._tmp_file = tmp.name
            cmd += ["--tags-file", self._tmp_file]

        self._task = asyncio.create_task(self._run(cmd, cwd=project_root))

    def _cleanup_tmp(self):
        if self._tmp_file and os.path.exists(self._tmp_file):
            try:
                os.unlink(self._tmp_file)
            except OSError:
                pass
            self._tmp_file = None

    async def stop(self):
        await super().stop()
        self._cleanup_tmp()
