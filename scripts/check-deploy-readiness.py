#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))

from app.core.config import get_settings
from app.core.local_db import LocalDatabase
from app.services.experiment_service import ExperimentService
from app.services.readiness import collect_readiness


STATUS_MARKS = {
    "ok": "PASS",
    "warn": "WARN",
    "error": "FAIL",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check WM Bench deployment readiness.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    settings = get_settings()
    service = ExperimentService(
        database=LocalDatabase(settings.database_path),
        resources_root=settings.resources_root,
        runs_root=settings.runs_root,
    )
    report = collect_readiness(settings, service)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"WM Bench deployment readiness: {report['status']}")
        print(f"Environment: {report['environment']}  Device: {report['device']}")
        for check in report["checks"]:
            mark = STATUS_MARKS.get(check["status"], check["status"].upper())
            required = "required" if check["required"] else "optional"
            print(f"[{mark}] {check['label']} ({required})")
            print(f"       {check['detail']}")

    return 1 if report["status"] == "not_ready" else 0


if __name__ == "__main__":
    raise SystemExit(main())
