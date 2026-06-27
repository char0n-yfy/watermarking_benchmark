from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))

from app.core.config import get_settings
from app.core.local_db import LocalDatabase
from app.services.experiment_service import ExperimentService


class RunTimeoutError(TimeoutError):
    pass


def build_service() -> ExperimentService:
    settings = get_settings()
    return ExperimentService(
        database=LocalDatabase(settings.database_path),
        resources_root=settings.resources_root,
        runs_root=settings.runs_root,
    )


def default_worker_id() -> str:
    return os.getenv("WM_BENCH_WORKER_ID") or f"{socket.gethostname()}-{os.getpid()}"


def _heartbeat(
    service: ExperimentService,
    *,
    worker_id: str,
    status: str,
    device: str,
    current_run_id: str | None = None,
    message: str | None = None,
) -> None:
    service.update_worker_heartbeat(
        worker_id=worker_id,
        status=status,
        pid=os.getpid(),
        device=device,
        current_run_id=current_run_id,
        message=message,
    )


def _raise_timeout(_signum: int, _frame: object) -> None:
    raise RunTimeoutError("run exceeded WM_BENCH_RUN_TIMEOUT_SECONDS")


def run_once(worker_id: str | None = None) -> int:
    settings = get_settings()
    service = build_service()
    resolved_worker_id = worker_id or default_worker_id()
    device = settings.device
    _heartbeat(service, worker_id=resolved_worker_id, status="idle", device=device)

    run = service.claim_next_run(resolved_worker_id)
    if run is None:
        return 0

    run_id = run["id"]
    log_path = Path(run["artifactRoot"]) / "worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _heartbeat(
        service,
        worker_id=resolved_worker_id,
        status="running",
        device=device,
        current_run_id=run_id,
        message=f"executing {run_id}",
    )

    try:
        with log_path.open("a", encoding="utf-8") as log_file:
            with redirect_stdout(log_file), redirect_stderr(log_file):
                print(f"[worker] start run={run_id} worker={resolved_worker_id} device={device}", flush=True)
                started = time.perf_counter()
                signal.signal(signal.SIGALRM, _raise_timeout)
                signal.alarm(max(0, settings.run_timeout_seconds))
                try:
                    result = service.execute_run(
                        run_id,
                        worker_id=resolved_worker_id,
                        device=device,
                        log_path=log_path,
                    )
                finally:
                    signal.alarm(0)
                elapsed = time.perf_counter() - started
                print(
                    f"[worker] finish run={run_id} status={result['status']} elapsed={elapsed:.2f}s",
                    flush=True,
                )
    except Exception as exc:
        _heartbeat(
            service,
            worker_id=resolved_worker_id,
            status="error",
            device=device,
            current_run_id=run_id,
            message=f"{type(exc).__name__}: {exc}",
        )
        raise

    _heartbeat(service, worker_id=resolved_worker_id, status="idle", device=device)
    return 1


def run_forever(poll_seconds: float, worker_id: str | None = None) -> None:
    settings = get_settings()
    service = build_service()
    resolved_worker_id = worker_id or default_worker_id()
    while True:
        try:
            processed = run_once(resolved_worker_id)
            if processed == 0:
                _heartbeat(
                    service,
                    worker_id=resolved_worker_id,
                    status="idle",
                    device=settings.device,
                    message="waiting for queued runs",
                )
        except Exception as exc:
            _heartbeat(
                service,
                worker_id=resolved_worker_id,
                status="error",
                device=settings.device,
                message=f"{type(exc).__name__}: {exc}",
            )
            print(f"[worker] error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        time.sleep(poll_seconds)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Local single-process watermark benchmark worker.")
    parser.add_argument("--once", action="store_true", help="Process one queued run and exit.")
    parser.add_argument("--poll-seconds", type=float, default=settings.worker_poll_seconds)
    parser.add_argument("--worker-id", default=default_worker_id())
    args = parser.parse_args()

    if args.once:
        processed = run_once(args.worker_id)
        print(f"processed {processed} queued run(s)")
        return
    run_forever(args.poll_seconds, args.worker_id)


if __name__ == "__main__":
    main()
