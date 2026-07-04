from __future__ import annotations

import contextvars
import os
from contextlib import contextmanager
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image

from evaluator.execution import ResolvedInt, resolve_cpu_workers
from evaluator.image_io import save_png_image


_CURRENT_SAVE_POOL: contextvars.ContextVar[AsyncImageSavePool | None] = contextvars.ContextVar(
    "wm_bench_current_image_save_pool",
    default=None,
)
_CURRENT_PREFETCHER: contextvars.ContextVar[ImageBatchPrefetcher | None] = contextvars.ContextVar(
    "wm_bench_current_image_batch_prefetcher",
    default=None,
)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _env_positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None and raw.strip() != "" else int(default)
    except (TypeError, ValueError):
        value = int(default)
    return max(1, value)


def load_rgb_image(path: str | Path) -> Image.Image:
    with Image.open(path) as opened:
        return opened.convert("RGB")


def _image_file_key(path: Path) -> tuple[str, int | None, int | None]:
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    try:
        stat = resolved.stat()
    except OSError:
        return (str(resolved), None, None)
    return (str(resolved), int(stat.st_mtime_ns), int(stat.st_size))


def _load_rgb_batch_direct(
    path_list: list[Path],
    *,
    workers_env: str,
    default_cap: int,
) -> list[Image.Image]:
    worker_config = resolve_cpu_workers(workers_env, len(path_list), enabled=True, default_cap=default_cap)
    if worker_config.value <= 1 or len(path_list) <= 1:
        return [load_rgb_image(path) for path in path_list]
    with ThreadPoolExecutor(max_workers=worker_config.value) as executor:
        return list(executor.map(load_rgb_image, path_list))


def load_rgb_batch(
    paths: Iterable[str | Path],
    *,
    workers_env: str = "WM_BENCH_IMAGE_IO_WORKERS",
    default_cap: int = 8,
) -> list[Image.Image]:
    path_list = [Path(path) for path in paths]
    prefetcher = current_image_batch_prefetcher()
    if prefetcher is not None:
        prefetched = prefetcher.take(path_list)
        if prefetched is not None:
            return prefetched
    return _load_rgb_batch_direct(path_list, workers_env=workers_env, default_cap=default_cap)


def _move_tensor_to_device(tensor: Any, device: Any) -> Any:
    non_blocking = False
    if str(device).startswith("cuda"):
        try:
            tensor = tensor.pin_memory()
            non_blocking = True
        except Exception:
            non_blocking = False
    return tensor.to(device, non_blocking=non_blocking)


def to_tensor_batch(
    images: Iterable[Image.Image],
    *,
    transform: Callable[[Image.Image], Any] | None = None,
    torch_module: Any | None = None,
    device: Any | None = None,
) -> Any:
    if torch_module is None:
        import torch as torch_module
    if transform is None:
        from torchvision.transforms import functional as TF

        transform = TF.to_tensor
    batch = torch_module.stack([transform(image) for image in images], dim=0)
    if device is not None:
        return _move_tensor_to_device(batch, device)
    return batch


class AsyncImageSavePool:
    """Per-stage image save pool.

    Batch implementations still call save_image_batch(). When a pool is active,
    those saves are submitted to a bounded stage-level executor and flushed by
    the runner before any code reads output metadata.
    """

    def __init__(
        self,
        *,
        workers_env: str = "WM_BENCH_IMAGE_IO_SAVE_WORKERS",
        default_cap: int = 8,
        enabled: bool | None = None,
    ) -> None:
        self.workers_env = workers_env
        self.default_cap = int(default_cap)
        self.enabled = _env_flag("WM_BENCH_ASYNC_IMAGE_SAVE", True) if enabled is None else bool(enabled)
        self.worker_config: ResolvedInt = resolve_cpu_workers(
            workers_env,
            1_000_000,
            enabled=self.enabled,
            default_cap=default_cap,
        )
        self.max_pending_limit = _env_positive_int(
            "WM_BENCH_ASYNC_IMAGE_SAVE_MAX_PENDING",
            max(1, self.worker_config.value * 4),
        )
        self._executor: ThreadPoolExecutor | None = None
        self._futures: list[Any] = []
        self._token: contextvars.Token[AsyncImageSavePool | None] | None = None
        self.submitted = 0
        self.flushed = 0
        self.max_pending = 0

    @property
    def active(self) -> bool:
        return bool(self.enabled and self.worker_config.value > 0)

    def __enter__(self) -> AsyncImageSavePool:
        if self.active:
            self._executor = ThreadPoolExecutor(max_workers=self.worker_config.value)
            self._token = _CURRENT_SAVE_POOL.set(self)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        flush_error: BaseException | None = None
        try:
            if self.active:
                self.flush()
        except BaseException as error:
            flush_error = error
        finally:
            if self._token is not None:
                _CURRENT_SAVE_POOL.reset(self._token)
                self._token = None
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None
        if exc_type is None and flush_error is not None:
            raise flush_error

    def submit(
        self,
        item: tuple[Image.Image, Path],
        save_fn: Callable[[Image.Image, Path], None],
    ) -> bool:
        if self._executor is None:
            return False
        image, path = item
        path.parent.mkdir(parents=True, exist_ok=True)
        future = self._executor.submit(save_fn, image, path)
        self._futures.append(future)
        self.submitted += 1
        self.max_pending = max(self.max_pending, len(self._futures))
        self._drain_until_below_limit()
        return True

    def _drain_until_below_limit(self) -> None:
        while len(self._futures) > self.max_pending_limit:
            future = self._futures.pop(0)
            future.result()
            self.flushed += 1

    def flush(self) -> None:
        futures, self._futures = self._futures, []
        for future in futures:
            future.result()
            self.flushed += 1

    def profile(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "active": bool(self.active),
            "workers": self.worker_config.to_json(),
            "maxPendingLimit": self.max_pending_limit,
            "submitted": self.submitted,
            "flushed": self.flushed,
            "maxPending": self.max_pending,
        }


def current_image_save_pool() -> AsyncImageSavePool | None:
    pool = _CURRENT_SAVE_POOL.get()
    if pool is not None and pool.active:
        return pool
    return None


class ImageBatchPrefetcher:
    """Single-stage decoded RGB image prefetcher for wrappers using load_rgb_batch()."""

    def __init__(
        self,
        *,
        workers_env: str = "WM_BENCH_IMAGE_IO_WORKERS",
        default_cap: int = 8,
        enabled: bool | None = None,
    ) -> None:
        self.workers_env = workers_env
        self.default_cap = int(default_cap)
        self.enabled = _env_flag("WM_BENCH_IMAGE_PREFETCH", True) if enabled is None else bool(enabled)
        self._executor: ThreadPoolExecutor | None = None
        self._futures: dict[tuple[tuple[str, int | None, int | None], ...], Future[list[Image.Image]]] = {}
        self._token: contextvars.Token[ImageBatchPrefetcher | None] | None = None
        self.submitted = 0
        self.consumed = 0
        self.hits = 0
        self.misses = 0

    @property
    def active(self) -> bool:
        return bool(self.enabled)

    def __enter__(self) -> ImageBatchPrefetcher:
        if self.active:
            self._executor = ThreadPoolExecutor(max_workers=1)
            self._token = _CURRENT_PREFETCHER.set(self)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            for future in list(self._futures.values()):
                try:
                    future.result()
                except Exception:
                    if exc_type is None:
                        raise
        finally:
            self._futures.clear()
            if self._token is not None:
                _CURRENT_PREFETCHER.reset(self._token)
                self._token = None
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None

    def _key(self, paths: Iterable[str | Path]) -> tuple[tuple[str, int | None, int | None], ...]:
        return tuple(_image_file_key(Path(path)) for path in paths)

    def prefetch(self, paths: Iterable[str | Path]) -> None:
        path_list = [Path(path) for path in paths]
        if not path_list or self._executor is None:
            return
        key = self._key(path_list)
        if key in self._futures:
            return
        self._futures[key] = self._executor.submit(
            _load_rgb_batch_direct,
            path_list,
            workers_env=self.workers_env,
            default_cap=self.default_cap,
        )
        self.submitted += 1

    def take(self, paths: Iterable[str | Path]) -> list[Image.Image] | None:
        key = self._key(paths)
        future = self._futures.pop(key, None)
        if future is None:
            self.misses += 1
            return None
        self.hits += 1
        self.consumed += 1
        return future.result()

    def profile(self) -> dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "active": bool(self.active and self._executor is not None),
            "submitted": self.submitted,
            "consumed": self.consumed,
            "hits": self.hits,
            "misses": self.misses,
            "pending": len(self._futures),
        }


def current_image_batch_prefetcher() -> ImageBatchPrefetcher | None:
    prefetcher = _CURRENT_PREFETCHER.get()
    if prefetcher is not None and prefetcher.active:
        return prefetcher
    return None


def prefetch_rgb_batch(paths: Iterable[str | Path]) -> None:
    prefetcher = current_image_batch_prefetcher()
    if prefetcher is not None:
        prefetcher.prefetch(paths)


@contextmanager
def suspend_image_save_pool():
    token = _CURRENT_SAVE_POOL.set(None)
    try:
        yield
    finally:
        _CURRENT_SAVE_POOL.reset(token)


def save_image_batch(
    items: Iterable[tuple[Image.Image, str | Path]],
    *,
    workers_env: str = "WM_BENCH_IMAGE_IO_SAVE_WORKERS",
    default_cap: int = 8,
    save_fn: Callable[[Image.Image, Path], None] | None = None,
) -> None:
    item_list = [(image, Path(path)) for image, path in items]
    save = save_fn or save_png_image
    pool = current_image_save_pool()

    def save_one(item: tuple[Image.Image, Path]) -> None:
        image, path = item
        path.parent.mkdir(parents=True, exist_ok=True)
        save(image, path)

    if pool is not None:
        for item in item_list:
            pool.submit(item, save)
        return

    worker_config = resolve_cpu_workers(workers_env, len(item_list), enabled=True, default_cap=default_cap)
    if worker_config.value <= 1 or len(item_list) <= 1:
        for item in item_list:
            save_one(item)
        return
    with ThreadPoolExecutor(max_workers=worker_config.value) as executor:
        list(executor.map(save_one, item_list))
