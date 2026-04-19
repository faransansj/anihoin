"""ExportJob — quantize_fp16.py / export_onnx.py subprocess 래퍼."""

import asyncio
import sys
from .base_job import BaseJob


class Fp16Job(BaseJob):
    def __init__(self):
        super().__init__("fp16")

    async def start(self, project_root: str):
        if self.state == "running":
            return
        self.state = "running"
        cmd = [sys.executable, "quantize_fp16.py"]
        self._task = asyncio.create_task(self._run(cmd, cwd=project_root))


class OnnxJob(BaseJob):
    def __init__(self):
        super().__init__("onnx")

    async def start(self, opset: int, project_root: str):
        if self.state == "running":
            return
        self.state = "running"
        cmd = [sys.executable, "export_onnx.py", "--opset", str(opset)]
        self._task = asyncio.create_task(self._run(cmd, cwd=project_root))
