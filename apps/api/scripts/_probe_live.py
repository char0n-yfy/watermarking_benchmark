import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from app.services.system_metrics import _lhm_power_from_json_tree, _LHM_POWER_SENSOR_NAMES

preferred = tuple(name.lower() for name in _LHM_POWER_SENSOR_NAMES)

try:
    api_resp = httpx.get("http://127.0.0.1:8000/system/metrics", timeout=5)
    print("API status:", api_resp.status_code)
    if api_resp.status_code == 200:
        print("API cpu:", api_resp.json().get("cpu"))
except Exception as exc:
    print("API error:", exc)

lhm = httpx.get("http://127.0.0.1:8085/data.json", timeout=5).json()
print("LHM parsed CPU power:", _lhm_power_from_json_tree(lhm, preferred))
