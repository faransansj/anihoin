"""ExportJob — quantize.py / export_onnx.py subprocess 래퍼."""

import asyncio
import sys
from .base_job import BaseJob

QUANT_FORMATS = ("fp16", "int8", "int4", "int2")


class QuantJob(BaseJob):
    def __init__(self):
        super().__init__("quant")

    async def start(self, fmt: str, project_root: str):
        if fmt not in QUANT_FORMATS:
            raise ValueError(f"Unknown format: {fmt}")
        if self.state == "running":
            return
        self.state = "running"
        cmd = [sys.executable, "quantize.py", "--format", fmt]
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
