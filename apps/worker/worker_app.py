from __future__ import annotations


def worker_profile() -> dict[str, str]:
    return {
        "profile": "local",
        "queue": "sqlite",
        "entrypoint": "python apps/worker/local_worker.py",
        "note": "Celery is intentionally not required for the local/AutoDL profile.",
    }
