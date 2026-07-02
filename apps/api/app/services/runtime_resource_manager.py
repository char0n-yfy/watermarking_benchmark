from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any, Callable

from app.services.experiment_schema import RUNTIME_PROFILE_SCHEMA


JsonDict = dict[str, Any]


class RuntimeResourceManager:
    def __init__(
        self,
        *,
        paths: dict[str, Path],
        run_id: str,
        device: str,
        append_jsonl: Callable[[Path, JsonDict], None],
        stage_event: Callable[..., None],
    ) -> None:
        self.paths = paths
        self.run_id = run_id
        self.device = device
        self.append_jsonl = append_jsonl
        self.stage_event = stage_event

    def cleanup(
        self,
        *,
        scope: str,
        reason: str,
        cell_key: str,
        release_watermarks: bool = False,
        release_attacks: bool = False,
        release_perceptual: bool = False,
        release_auxiliary: bool = False,
        metadata: JsonDict | None = None,
    ) -> JsonDict:
        started = time.perf_counter()
        before = self._snapshot()
        actions: list[str] = []
        errors: list[str] = []

        def run_action(name: str, fn: Callable[[], None]) -> None:
            actions.append(name)
            try:
                fn()
            except Exception as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")

        if release_watermarks:
            run_action("clear_watermark_cache", self._clear_watermark_cache)
        if release_attacks:
            run_action("clear_attack_cache", self._clear_attack_cache)
        if release_perceptual:
            run_action("clear_perceptual_backend", self._clear_perceptual_backend)
        if release_auxiliary:
            run_action("clear_auxiliary_model_caches", self._clear_auxiliary_model_caches)

        run_action("gc_collect", gc.collect)
        run_action("torch_cleanup", self._torch_cleanup)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        after = self._snapshot()
        status = "succeeded" if not errors else "partial"
        record = RUNTIME_PROFILE_SCHEMA.apply(
            {
                "runId": self.run_id,
                "cellKey": cell_key,
                "stage": "resource_cleanup",
                "method": scope,
                "device": self.device,
                "status": status,
                "imageCount": 0,
                "totalMegapixels": 0.0,
                "elapsedMs": elapsed_ms,
                "peakMemoryMB": after.get("cuda", {}).get("maxAllocatedMB") or after.get("rssMB"),
                "peakMemorySource": "cleanup_snapshot",
                "error": "; ".join(errors) if errors else None,
                "metadata": {
                    "scope": scope,
                    "reason": reason,
                    "actions": actions,
                    "errors": errors,
                    "before": before,
                    "after": after,
                    **(metadata or {}),
                },
                "timestamp": self._utc_timestamp(),
            }
        )
        self.append_jsonl(self.paths["runtimeProfile"], record)
        self.stage_event(
            self.paths,
            self.run_id,
            "resource_cleanup",
            status,
            scope=scope,
            reason=reason,
            cellKey=cell_key,
            actions=actions,
            errors=errors,
            elapsedMs=elapsed_ms,
            before=before,
            after=after,
            **(metadata or {}),
        )
        return record

    def _clear_watermark_cache(self) -> None:
        from evaluator.watermarking.runner import clear_watermark_cache

        clear_watermark_cache()

    def _clear_attack_cache(self) -> None:
        from evaluator.attacks.runner import clear_attack_cache

        clear_attack_cache()

    def _clear_perceptual_backend(self) -> None:
        from app.services.scoring import clear_perceptual_backend

        clear_perceptual_backend()

    def _clear_auxiliary_model_caches(self) -> None:
        try:
            from evaluator.attacks.consumer_enhancement_workflow_attacks.backends import deep_enhance

            deep_enhance.clear_model_cache()
        except Exception:
            pass
        try:
            from evaluator.attacks.consumer_enhancement_workflow_attacks.backends import restoration_sr

            restoration_sr.clear_model_cache()
        except Exception:
            pass

    def _torch_cleanup(self) -> None:
        from evaluator.runtime_cleanup import torch_cleanup

        torch_cleanup(reset_peak=True)

    def _snapshot(self) -> JsonDict:
        snapshot: JsonDict = {
            "rssMB": self._current_rss_mb(),
        }
        cuda = self._cuda_snapshot()
        if cuda:
            snapshot["cuda"] = cuda
        return snapshot

    @staticmethod
    def _current_rss_mb() -> float | None:
        status_path = Path("/proc/self/status")
        try:
            for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
        except Exception:
            return None
        return None

    def _cuda_snapshot(self) -> JsonDict:
        if not str(self.device).startswith("cuda"):
            return {}
        try:
            import torch

            if not torch.cuda.is_available():
                return {}
            torch_device = torch.device(self.device)
            with torch.cuda.device(torch_device):
                torch.cuda.synchronize(torch_device)
                free_bytes, total_bytes = torch.cuda.mem_get_info(torch_device)
                return {
                    "device": str(torch_device),
                    "allocatedMB": float(torch.cuda.memory_allocated(torch_device)) / (1024.0 * 1024.0),
                    "reservedMB": float(torch.cuda.memory_reserved(torch_device)) / (1024.0 * 1024.0),
                    "maxAllocatedMB": float(torch.cuda.max_memory_allocated(torch_device)) / (1024.0 * 1024.0),
                    "maxReservedMB": float(torch.cuda.max_memory_reserved(torch_device)) / (1024.0 * 1024.0),
                    "freeMB": float(free_bytes) / (1024.0 * 1024.0),
                    "totalMB": float(total_bytes) / (1024.0 * 1024.0),
                }
        except Exception:
            return {}

    @staticmethod
    def _utc_timestamp() -> str:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat()
