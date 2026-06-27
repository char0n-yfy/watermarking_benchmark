from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from evaluator.watermarking.base import BaseWatermark, WatermarkContext
from evaluator.watermarking.registry import register_watermark
from evaluator.watermarking.utils import (
    packaged_algorithm_dir,
    prepend_sys_path,
    require_path,
)


@register_watermark
class BlindWatermark(BaseWatermark):
    name = "blind_watermark"
    description = "DWT-DCT-SVD blind watermark wrapper with packaged source and no weights."

    def __init__(
        self,
        repo_dir: str | Path | None = None,
        password_img: int = 1,
        password_wm: int = 1,
        **params: Any,
    ) -> None:
        super().__init__(
            repo_dir=str(repo_dir) if repo_dir is not None else None,
            password_img=password_img,
            password_wm=password_wm,
            **params,
        )
        self.repo_dir = require_path(repo_dir or packaged_algorithm_dir("blind_watermark"), "blind_watermark repo_dir")
        self.password_img = int(password_img)
        self.password_wm = int(password_wm)

    def _message(self, context: WatermarkContext) -> str:
        return context.message or "test001"

    @staticmethod
    def _wm_shape(message: str) -> int:
        if not message:
            return 1
        return len(bin(int(message.encode("utf-8").hex(), base=16))[2:])

    def embed_impl(self, input_path: Path, output_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        with prepend_sys_path(self.repo_dir, ["blind_watermark"]):
            from blind_watermark import WaterMark

            bwm = WaterMark(password_img=self.password_img, password_wm=self.password_wm)
            with tempfile.NamedTemporaryFile(prefix="blind_watermark_input_", suffix=input_path.suffix or ".png", delete=False) as src_tmp:
                src_tmp_path = Path(src_tmp.name)
            with tempfile.NamedTemporaryFile(prefix="blind_watermark_output_", suffix=".png", delete=False) as out_tmp:
                tmp_output = Path(out_tmp.name)
            try:
                shutil.copyfile(input_path, src_tmp_path)
                bwm.read_img(filename=str(src_tmp_path))
                bwm.read_wm(self._message(context), mode="str")
                bwm.embed(str(tmp_output))
                shutil.copyfile(tmp_output, output_path)
            finally:
                src_tmp_path.unlink(missing_ok=True)
                tmp_output.unlink(missing_ok=True)

        return {
            "message": self._message(context),
            "wm_shape": self._wm_shape(self._message(context)),
            "checkpoint_file": None,
        }

    def extract_impl(self, input_path: Path, context: WatermarkContext) -> Mapping[str, Any]:
        message = self._message(context)
        with prepend_sys_path(self.repo_dir, ["blind_watermark"]):
            from blind_watermark import WaterMark

            bwm = WaterMark(password_img=self.password_img, password_wm=self.password_wm)
            with tempfile.NamedTemporaryFile(prefix="blind_watermark_decode_", suffix=input_path.suffix or ".png", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                shutil.copyfile(input_path, tmp_path)
                decoded = str(bwm.extract(str(tmp_path), wm_shape=self._wm_shape(message), mode="str"))
            finally:
                tmp_path.unlink(missing_ok=True)

        return {
            "message": decoded,
            "expected_message": message,
            "match": decoded == message,
            "wm_shape": self._wm_shape(message),
            "checkpoint_file": None,
        }
