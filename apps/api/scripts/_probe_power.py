import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.services.system_metrics import (
    _windows_cpu_power_draw_w,
    _cpu_power_draw_w,
    collect_system_metrics,
)

print("raw power:", _windows_cpu_power_draw_w())
print("smoothed power:", _cpu_power_draw_w())
payload = collect_system_metrics(data_root=Path("."), device="cpu")
print("cpu:", payload["cpu"])
