from __future__ import annotations

import gc
from typing import Any


def move_to_cpu(obj: Any) -> None:
    """Best-effort move for torch modules/pipelines before dropping references."""
    try:
        to_fn = getattr(obj, "to", None)
        if callable(to_fn):
            to_fn("cpu")
    except Exception:
        pass


def torch_cleanup(*, reset_peak: bool = False) -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
            if reset_peak:
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
    except Exception:
        pass
