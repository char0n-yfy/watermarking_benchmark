from __future__ import annotations

import os
import platform
import re
import shutil

try:
    import httpx
except ModuleNotFoundError:
    httpx = None  # type: ignore[assignment]

try:
    import resource
except ModuleNotFoundError:
    resource = None  # Windows has no resource module

try:
    import psutil
except ModuleNotFoundError:
    psutil = None  # type: ignore[assignment]

try:
    import win32pdh
except ModuleNotFoundError:
    win32pdh = None  # type: ignore[assignment]

import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_DISK_IO_SAMPLE: tuple[float, int, int] | None = None
_RAPL_ENERGY_SAMPLE: tuple[float, int] | None = None
_CPU_POWER_W_EMA: float | None = None
_CPU_TEMP_C_EMA: float | None = None
_LHM_HTTP_CLIENT: Any = None
_LHM_HTTP_PAYLOAD_CACHE: tuple[float, dict[str, Any]] | None = None
_LHM_POWER_RAW_SAMPLES: list[float] = []
_WINDOWS_PDH_POWER: dict[str, Any] = {"query": None, "counter": None, "primed": False}
_WINDOWS_PDH_SAMPLES: list[float] = []
_MAX_PDH_POWER_SAMPLES = 5
_MAX_LHM_POWER_SAMPLES = 3
_LHM_HTTP_CACHE_TTL_S = 0.5
_LHM_POWER_NAMESPACES = (
    r"root\LibreHardwareMonitor",
    r"root\OpenHardwareMonitor",
)
_LHM_POWER_SENSOR_NAMES = (
    "CPU Package",
    "Processor Package",
    "CPU Power",
    "Package",
)
_LHM_POWER_EXACT_NAMES = frozenset(
    name.lower() for name in ("CPU Package", "Processor Package")
)
_LHM_TEMPERATURE_SENSOR_NAMES = (
    "CPU Package",
    "Core Max",
    "Core Average",
    "Tctl",
    "Tdie",
)
_LHM_TEMPERATURE_EXACT_NAMES = frozenset(
    name.lower() for name in ("CPU Package", "Core Max")
)


def collect_system_metrics(*, data_root: Path, device: str) -> dict[str, Any]:
    disk = shutil.disk_usage(data_root)
    memory = _memory_metrics()
    cpu_percent = _cpu_usage_percent()
    uptime_seconds = _uptime_seconds()
    disk_io = _disk_io_rates()

    cpu_sensors = _cpu_sensor_metrics()
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostName": socket.gethostname(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "pythonVersion": platform.python_version(),
        "uptimeSeconds": uptime_seconds,
        "configuredDevice": device,
        "cpu": {
            "logicalCores": os.cpu_count() or 0,
            "usagePercent": cpu_percent,
            "loadAverage": _load_average(),
            "temperatureC": cpu_sensors["temperatureC"],
            "powerDrawW": cpu_sensors["powerDrawW"],
        },
        "memory": memory,
        "disk": {
            "path": str(data_root),
            "totalBytes": disk.total,
            "usedBytes": disk.used,
            "freeBytes": disk.free,
            "usedPercent": _percent(disk.used, disk.total),
            **disk_io,
        },
        "gpu": _gpu_metrics(),
        "process": {
            "pid": os.getpid(),
            "rssBytes": _process_rss_bytes(),
            "pythonExecutable": sys.executable,
        },
    }


def _percent(used: float, total: float) -> float | None:
    if total <= 0:
        return None
    return round((used / total) * 100, 2)


def _is_windows() -> bool:
    return platform.system() == "Windows"


def _load_average() -> list[float]:
    getloadavg = getattr(os, "getloadavg", None)
    if getloadavg is None:
        return []
    try:
        return [round(value, 2) for value in getloadavg()]
    except OSError:
        return []


def _read_proc_stat() -> tuple[int, int] | None:
    stat_path = Path("/proc/stat")
    if not stat_path.exists():
        return None
    fields = stat_path.read_text(encoding="utf-8").splitlines()[0].split()[1:]
    values = [int(value) for value in fields]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def _cpu_usage_percent() -> float | None:
    proc_start = _read_proc_stat()
    if proc_start is not None:
        time.sleep(0.1)
        proc_end = _read_proc_stat()
        if proc_end is not None:
            total_delta = proc_end[0] - proc_start[0]
            idle_delta = proc_end[1] - proc_start[1]
            if total_delta > 0:
                return round(max(0.0, min(100.0, (1 - idle_delta / total_delta) * 100)), 2)

    top_output = _run_command(["top", "-l", "1", "-n", "0"], timeout=5)
    match = re.search(r"CPU usage:.*?([\d.]+)% idle", top_output)
    if match:
        return round(max(0.0, min(100.0, 100.0 - float(match.group(1)))), 2)

    loads = _load_average()
    cores = os.cpu_count() or 1
    if loads:
        return round(max(0.0, min(100.0, (loads[0] / cores) * 100)), 2)

    windows_cpu = _windows_cpu_usage_percent()
    if windows_cpu is not None:
        return windows_cpu
    return None


def _cpu_sensor_metrics() -> dict[str, float | None]:
    return {
        "temperatureC": _cpu_temperature_c(),
        "powerDrawW": _cpu_power_draw_w(),
    }


def _cpu_temperature_c() -> float | None:
    if _is_windows():
        return _smooth_cpu_temp_c(_windows_cpu_temperature_c())
    for reader in (_linux_cpu_temperature_c, _psutil_cpu_temperature_c):
        value = reader()
        if value is not None:
            return value
    return None


def _linux_cpu_temperature_c() -> float | None:
    thermal_root = Path("/sys/class/thermal")
    if not thermal_root.exists():
        return None
    readings: list[float] = []
    for zone in thermal_root.glob("thermal_zone*/temp"):
        try:
            readings.append(int(zone.read_text(encoding="utf-8").strip()) / 1000.0)
        except (OSError, ValueError):
            continue
    if readings:
        return round(max(readings), 1)
    return None


def _psutil_cpu_temperature_c() -> float | None:
    if psutil is None or not hasattr(psutil, "sensors_temperatures"):
        return None
    try:
        sensors = psutil.sensors_temperatures(fahrenheit=False)
    except (AttributeError, OSError, RuntimeError):
        return None
    preferred = ("coretemp", "k10temp", "cpu_thermal", "acpitz", "cpu-thermal")
    for name in preferred:
        entries = sensors.get(name)
        if entries:
            return round(float(entries[0].current), 1)
    for entries in sensors.values():
        if entries:
            return round(float(entries[0].current), 1)
    return None


def _windows_cpu_temperature_c() -> float | None:
    if not _is_windows():
        return None

    lhm_temperature = _windows_lhm_http_cpu_temperature_c()
    if lhm_temperature is not None:
        return lhm_temperature

    high_precision = _run_powershell_counter(
        "(Get-Counter '\\Thermal Zone Information(*)\\High Precision Temperature' -ErrorAction SilentlyContinue)"
        ".CounterSamples | Sort-Object CookedValue -Descending | Select-Object -ExpandProperty CookedValue -First 1"
    )
    if high_precision is not None and high_precision > 0:
        return round((high_precision / 10.0) - 273.15, 1)

    temperature = _run_powershell_counter(
        "(Get-Counter '\\Thermal Zone Information(*)\\Temperature' -ErrorAction SilentlyContinue)"
        ".CounterSamples | Sort-Object CookedValue -Descending | Select-Object -ExpandProperty CookedValue -First 1"
    )
    if temperature is not None and temperature > 0:
        if temperature > 200:
            return round(temperature - 273.15, 1)
        return round(temperature, 1)

    output = _run_command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance -Namespace root/wmi -ClassName MSAcpi_ThermalZoneTemperature "
            "| Select-Object -ExpandProperty CurrentTemperature -First 1",
        ],
        timeout=5,
    )
    raw = _float_or_none(output.strip())
    if raw is None:
        return None
    return round((raw / 10.0) - 273.15, 1)


def _windows_cpu_power_draw_w() -> float | None:
    if not _is_windows():
        return None

    for reader in (
        _windows_lhm_http_cpu_power_w,
        _windows_hardware_monitor_cpu_power_w,
        _windows_pdh_cpu_power_w,
        _windows_powershell_cpu_power_w,
    ):
        watts = reader()
        if watts is not None:
            return watts
    return None


def _lhm_http_port() -> int:
    raw = os.environ.get("WM_BENCH_LHM_PORT", "8085")
    try:
        return int(raw)
    except ValueError:
        return 8085


def _parse_lhm_sensor_watts(raw: object) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return value if value > 0 else None
    text = str(raw).strip()
    if not text:
        return None
    match = re.search(r"([\d.]+)", text)
    if not match:
        return None
    value = float(match.group(1))
    return value if value > 0 else None


def _lhm_power_from_json_tree(node: object, preferred_names: tuple[str, ...]) -> float | None:
    if not isinstance(node, dict):
        return None

    exact_reading: float | None = None
    preferred_reading: float | None = None
    fallback_reading: float | None = None
    stack = [node]
    while stack:
        current = stack.pop()
        if not isinstance(current, dict):
            continue
        if str(current.get("Type", "")).lower() == "power":
            watts = _parse_lhm_sensor_watts(current.get("RawValue"))
            if watts is None:
                watts = _parse_lhm_sensor_watts(current.get("Value"))
            if watts is not None:
                name = str(current.get("Text", "")).lower()
                if name in _LHM_POWER_EXACT_NAMES:
                    exact_reading = watts if exact_reading is None else max(exact_reading, watts)
                elif any(token in name for token in preferred_names):
                    preferred_reading = watts if preferred_reading is None else max(preferred_reading, watts)
                else:
                    fallback_reading = watts if fallback_reading is None else max(fallback_reading, watts)
        for child in current.get("Children", []) or []:
            stack.append(child)

    reading = exact_reading if exact_reading is not None else preferred_reading
    if reading is None:
        reading = fallback_reading
    if reading is None:
        return None
    return round(reading, 1)


def _parse_lhm_sensor_celsius(raw: object) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        value = float(raw)
        return value if value > 0 else None
    text = str(raw).strip()
    if not text:
        return None
    match = re.search(r"([\d.]+)", text)
    if not match:
        return None
    value = float(match.group(1))
    return value if value > 0 else None


def _lhm_temperature_from_json_tree(node: object, preferred_names: tuple[str, ...]) -> float | None:
    if not isinstance(node, dict):
        return None

    exact_reading: float | None = None
    preferred_reading: float | None = None
    fallback_reading: float | None = None
    stack = [node]
    while stack:
        current = stack.pop()
        if not isinstance(current, dict):
            continue
        if str(current.get("Type", "")).lower() == "temperature":
            celsius = _parse_lhm_sensor_celsius(current.get("RawValue"))
            if celsius is None:
                celsius = _parse_lhm_sensor_celsius(current.get("Value"))
            if celsius is not None:
                name = str(current.get("Text", "")).lower()
                if name in _LHM_TEMPERATURE_EXACT_NAMES:
                    exact_reading = celsius if exact_reading is None else max(exact_reading, celsius)
                elif any(token in name for token in preferred_names):
                    preferred_reading = celsius if preferred_reading is None else max(preferred_reading, celsius)
                else:
                    fallback_reading = celsius if fallback_reading is None else max(fallback_reading, celsius)
        for child in current.get("Children", []) or []:
            stack.append(child)

    reading = exact_reading if exact_reading is not None else preferred_reading
    if reading is None:
        reading = fallback_reading
    if reading is None:
        return None
    return round(reading, 1)


def _fetch_lhm_http_payload() -> dict[str, Any] | None:
    global _LHM_HTTP_PAYLOAD_CACHE

    now = time.monotonic()
    if _LHM_HTTP_PAYLOAD_CACHE is not None:
        cached_at, payload = _LHM_HTTP_PAYLOAD_CACHE
        if now - cached_at < _LHM_HTTP_CACHE_TTL_S:
            return payload

    client = _get_lhm_http_client()
    if client is None:
        return None

    port = _lhm_http_port()
    try:
        response = client.get(f"http://127.0.0.1:{port}/data.json")
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    _LHM_HTTP_PAYLOAD_CACHE = (now, payload)
    return payload


def _median_lhm_power_w(raw: float) -> float:
    samples = _LHM_POWER_RAW_SAMPLES
    samples.append(raw)
    if len(samples) > _MAX_LHM_POWER_SAMPLES:
        del samples[0]
    return round(sorted(samples)[len(samples) // 2], 1)


def _get_lhm_http_client() -> Any:
    """Local LHM web server must bypass system HTTP proxies (otherwise 502)."""
    global _LHM_HTTP_CLIENT

    if httpx is None:
        return None
    if _LHM_HTTP_CLIENT is None:
        _LHM_HTTP_CLIENT = httpx.Client(trust_env=False, timeout=2.0)
    return _LHM_HTTP_CLIENT


def _windows_lhm_http_cpu_power_w() -> float | None:
    payload = _fetch_lhm_http_payload()
    if payload is None:
        return None

    preferred_names = tuple(name.lower() for name in _LHM_POWER_SENSOR_NAMES)
    raw = _lhm_power_from_json_tree(payload, preferred_names)
    if raw is None:
        return None
    return _median_lhm_power_w(raw)


def _windows_lhm_http_cpu_temperature_c() -> float | None:
    payload = _fetch_lhm_http_payload()
    if payload is None:
        return None

    preferred_names = tuple(name.lower() for name in _LHM_TEMPERATURE_SENSOR_NAMES)
    return _lhm_temperature_from_json_tree(payload, preferred_names)


def _windows_hardware_monitor_cpu_power_w() -> float | None:
    try:
        import win32com.client
    except ImportError:
        return None

    preferred_names = tuple(name.lower() for name in _LHM_POWER_SENSOR_NAMES)
    for namespace in _LHM_POWER_NAMESPACES:
        try:
            locator = win32com.client.Dispatch("WbemScripting.SWbemLocator")
            service = locator.ConnectServer(".", namespace)
            sensors = service.ExecQuery("SELECT Name, Value FROM Sensor WHERE SensorType='Power'")
        except Exception:
            continue

        preferred_reading: float | None = None
        fallback_reading: float | None = None
        for sensor in sensors:
            try:
                value = float(sensor.Value)
            except (AttributeError, TypeError, ValueError):
                continue
            if value <= 0:
                continue
            name = str(sensor.Name).lower()
            if any(token in name for token in preferred_names):
                preferred_reading = value if preferred_reading is None else max(preferred_reading, value)
                continue
            fallback_reading = value if fallback_reading is None else max(fallback_reading, value)

        reading = preferred_reading if preferred_reading is not None else fallback_reading
        if reading is not None:
            return round(reading, 1)
    return None


def _windows_pdh_cpu_power_w() -> float | None:
    if win32pdh is None:
        return None

    state = _WINDOWS_PDH_POWER
    try:
        if state["query"] is None:
            state["query"] = win32pdh.OpenQuery()
            state["counter"] = win32pdh.AddCounter(state["query"], r"\Power Meter(_Total)\Power")
            win32pdh.CollectQueryData(state["query"])
            time.sleep(0.05)
            win32pdh.CollectQueryData(state["query"])
            state["primed"] = True

        win32pdh.CollectQueryData(state["query"])
        _, milliwatts = win32pdh.GetFormattedCounterValue(state["counter"], win32pdh.PDH_FMT_DOUBLE)
        if milliwatts is None or milliwatts <= 0:
            return None
        watts = round(float(milliwatts) / 1000.0, 2)
        samples = _WINDOWS_PDH_SAMPLES
        samples.append(watts)
        if len(samples) > _MAX_PDH_POWER_SAMPLES:
            del samples[0]
        return round(sorted(samples)[len(samples) // 2], 1)
    except Exception:
        _reset_windows_pdh_power()
        return None


def _reset_windows_pdh_power() -> None:
    global _WINDOWS_PDH_SAMPLES

    state = _WINDOWS_PDH_POWER
    query = state["query"]
    state["query"] = None
    state["counter"] = None
    state["primed"] = False
    _WINDOWS_PDH_SAMPLES = []
    if query is not None and win32pdh is not None:
        try:
            win32pdh.CloseQuery(query)
        except Exception:
            pass


def _windows_powershell_cpu_power_w() -> float | None:
    milliwatts = _run_powershell_counter(
        "(Get-Counter '\\Power Meter(_Total)\\Power' -ErrorAction SilentlyContinue).CounterSamples "
        "| Select-Object -ExpandProperty CookedValue -First 1"
    )
    if milliwatts is None or milliwatts <= 0:
        milliwatts = _run_powershell_counter(
            "(Get-Counter '\\Power Meter(*)\\Power' -ErrorAction SilentlyContinue).CounterSamples "
            "| Sort-Object CookedValue -Descending | Select-Object -ExpandProperty CookedValue -First 1"
        )
    if milliwatts is None or milliwatts <= 0:
        return None
    return round(milliwatts / 1000.0, 1)


def _run_powershell_counter(command: str) -> float | None:
    output = _run_command(["powershell", "-NoProfile", "-Command", command], timeout=8)
    if not output.strip():
        return None
    return _first_float_in_output(output)


def _first_float_in_output(output: str) -> float | None:
    for line in output.splitlines():
        value = _float_or_none(line.strip())
        if value is not None:
            return value
    return None


def _cpu_power_draw_w() -> float | None:
    raw: float | None = None
    if _is_windows():
        raw = _windows_cpu_power_draw_w()
    else:
        raw = _linux_cpu_power_draw_w()
    return _smooth_cpu_power_w(raw)


def _smooth_cpu_power_w(raw: float | None) -> float | None:
    global _CPU_POWER_W_EMA

    if raw is None:
        return _CPU_POWER_W_EMA
    if _CPU_POWER_W_EMA is None:
        _CPU_POWER_W_EMA = raw
    else:
        _CPU_POWER_W_EMA = round((0.45 * raw) + (0.55 * _CPU_POWER_W_EMA), 2)
    return round(_CPU_POWER_W_EMA, 1)


def _smooth_cpu_temp_c(raw: float | None) -> float | None:
    global _CPU_TEMP_C_EMA

    if raw is None:
        return _CPU_TEMP_C_EMA
    if _CPU_TEMP_C_EMA is None:
        _CPU_TEMP_C_EMA = raw
    else:
        if abs(raw - _CPU_TEMP_C_EMA) >= 12.0:
            _CPU_TEMP_C_EMA = raw
        else:
            _CPU_TEMP_C_EMA = round((0.45 * raw) + (0.55 * _CPU_TEMP_C_EMA), 2)
    return round(_CPU_TEMP_C_EMA, 1)


def warmup_cpu_power_sensors(*, attempts: int = 10, delay_seconds: float = 0.4) -> None:
    """Prime sensor readers so the dashboard has power data on the first refresh."""
    if not _is_windows():
        time.sleep(delay_seconds)
        for _ in range(attempts):
            reading = _linux_cpu_power_draw_w()
            if reading is not None:
                _smooth_cpu_power_w(reading)
                return
            time.sleep(delay_seconds)
        return

    for attempt in range(attempts):
        reading = _windows_cpu_power_draw_w()
        if reading is not None:
            _smooth_cpu_power_w(reading)
            return
        if attempt < attempts - 1:
            time.sleep(delay_seconds)


def _read_rapl_energy_microjoules() -> int | None:
    rapl_root = Path("/sys/class/powercap/intel-rapl")
    if not rapl_root.exists():
        return None
    for package in sorted(rapl_root.glob("intel-rapl:*")):
        energy_path = package / "energy_uj"
        if not energy_path.exists():
            continue
        try:
            return int(energy_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
    return None


def _linux_cpu_power_draw_w() -> float | None:
    global _RAPL_ENERGY_SAMPLE

    energy = _read_rapl_energy_microjoules()
    if energy is None:
        return None
    now = time.monotonic()
    previous = _RAPL_ENERGY_SAMPLE
    _RAPL_ENERGY_SAMPLE = (now, energy)
    if previous is None:
        return None
    elapsed = now - previous[0]
    if elapsed <= 0:
        return None
    delta = energy - previous[1]
    if delta < 0:
        _RAPL_ENERGY_SAMPLE = None
        return None
    return round((delta / 1_000_000) / elapsed, 1)


def _memory_metrics() -> dict[str, int | float | None]:
    linux_memory = _linux_memory_metrics()
    if linux_memory is not None:
        return linux_memory
    darwin_memory = _darwin_memory_metrics()
    if darwin_memory is not None:
        return darwin_memory
    windows_memory = _windows_memory_metrics()
    if windows_memory is not None:
        return windows_memory
    return {
        "totalBytes": None,
        "usedBytes": None,
        "availableBytes": None,
        "usedPercent": None,
    }


def _linux_memory_metrics() -> dict[str, int | float | None] | None:
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return None
    values: dict[str, int] = {}
    for line in meminfo_path.read_text(encoding="utf-8").splitlines():
        key, raw_value = line.split(":", 1)
        values[key] = int(raw_value.strip().split()[0]) * 1024
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if total is None or available is None:
        return None
    used = total - available
    return {
        "totalBytes": total,
        "usedBytes": used,
        "availableBytes": available,
        "usedPercent": _percent(used, total),
    }


def _darwin_memory_metrics() -> dict[str, int | float | None] | None:
    if platform.system() != "Darwin":
        return None
    total_output = _run_command(["sysctl", "-n", "hw.memsize"], timeout=5).strip()
    vm_output = _run_command(["vm_stat"], timeout=5)
    if not total_output or not vm_output:
        return None
    total = int(total_output)
    page_size_match = re.search(r"page size of (\d+) bytes", vm_output)
    page_size = int(page_size_match.group(1)) if page_size_match else 4096
    pages: dict[str, int] = {}
    for line in vm_output.splitlines():
        match = re.match(r"Pages ([\w ]+):\s+(\d+).", line.strip())
        if match:
            pages[match.group(1).strip()] = int(match.group(2))
    available_pages = (
        pages.get("free", 0)
        + pages.get("inactive", 0)
        + pages.get("speculative", 0)
        + pages.get("purgeable", 0)
    )
    available = min(total, available_pages * page_size)
    used = max(0, total - available)
    return {
        "totalBytes": total,
        "usedBytes": used,
        "availableBytes": available,
        "usedPercent": _percent(used, total),
    }


def _windows_memory_metrics() -> dict[str, int | float | None] | None:
    if not _is_windows() or psutil is None:
        return None
    virtual = psutil.virtual_memory()
    return {
        "totalBytes": int(virtual.total),
        "usedBytes": int(virtual.used),
        "availableBytes": int(virtual.available),
        "usedPercent": _percent(float(virtual.used), float(virtual.total)),
    }


def _windows_cpu_usage_percent() -> float | None:
    if not _is_windows() or psutil is None:
        return None
    value = psutil.cpu_percent(interval=0.1)
    return round(max(0.0, min(100.0, float(value))), 2)


def _nvidia_smi_executable() -> str:
    if _is_windows():
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        bundled = Path(program_files) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
        if bundled.is_file():
            return str(bundled)
    return "nvidia-smi"


def _gpu_metrics() -> dict[str, Any]:
    output = _run_command(
        [
            _nvidia_smi_executable(),
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ],
        timeout=5,
    )
    devices = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 7:
            continue
        memory_used = _float_or_none(parts[3])
        memory_total = _float_or_none(parts[4])
        devices.append(
            {
                "index": int(float(parts[0])),
                "name": parts[1],
                "utilizationPercent": _float_or_none(parts[2]),
                "memoryUsedMiB": memory_used,
                "memoryTotalMiB": memory_total,
                "memoryUsedPercent": _percent(memory_used or 0, memory_total or 0),
                "temperatureC": _float_or_none(parts[5]),
                "powerDrawW": _float_or_none(parts[6]),
            }
        )
    return {"available": bool(devices), "devices": devices}


def _disk_io_rates() -> dict[str, float | None]:
    linux_rates = _linux_disk_io_rates()
    if linux_rates is not None:
        return linux_rates
    darwin_rates = _darwin_disk_io_rates()
    if darwin_rates is not None:
        return darwin_rates
    windows_rates = _windows_disk_io_rates()
    if windows_rates is not None:
        return windows_rates
    return {
        "ioReadBytesPerSecond": None,
        "ioWriteBytesPerSecond": None,
        "ioTotalBytesPerSecond": None,
    }


def _disk_io_rates_from_samples(
    *,
    now: float,
    read_bytes: int,
    write_bytes: int,
) -> dict[str, float | None]:
    global _DISK_IO_SAMPLE

    previous = _DISK_IO_SAMPLE
    _DISK_IO_SAMPLE = (now, read_bytes, write_bytes)
    if previous is None:
        return {
            "ioReadBytesPerSecond": None,
            "ioWriteBytesPerSecond": None,
            "ioTotalBytesPerSecond": None,
        }

    elapsed = now - previous[0]
    if elapsed <= 0:
        return {
            "ioReadBytesPerSecond": None,
            "ioWriteBytesPerSecond": None,
            "ioTotalBytesPerSecond": None,
        }

    read_rate = max(0.0, (read_bytes - previous[1]) / elapsed)
    write_rate = max(0.0, (write_bytes - previous[2]) / elapsed)
    return {
        "ioReadBytesPerSecond": round(read_rate, 2),
        "ioWriteBytesPerSecond": round(write_rate, 2),
        "ioTotalBytesPerSecond": round(read_rate + write_rate, 2),
    }


def _linux_disk_io_rates() -> dict[str, float | None] | None:
    stat_root = Path("/sys/block")
    if not stat_root.exists():
        return None

    read_bytes = 0
    write_bytes = 0
    found = False
    for device in stat_root.iterdir():
        if device.name.startswith(("loop", "ram", "fd")):
            continue
        stat_path = device / "stat"
        if not stat_path.exists():
            continue
        fields = stat_path.read_text(encoding="utf-8").split()
        if len(fields) < 7:
            continue
        read_bytes += int(fields[2]) * 512
        write_bytes += int(fields[6]) * 512
        found = True

    if not found:
        return None

    return _disk_io_rates_from_samples(
        now=time.monotonic(),
        read_bytes=read_bytes,
        write_bytes=write_bytes,
    )


def _darwin_disk_io_rates() -> dict[str, float | None] | None:
    if platform.system() != "Darwin":
        return None

    output = _run_command(["iostat", "-d", "-w", "1", "-c", "2"], timeout=4)
    numeric_rows: list[list[str]] = []
    for line in output.splitlines():
        parts = line.split()
        if parts and _float_or_none(parts[0]) is not None:
            numeric_rows.append(parts)
    if not numeric_rows:
        return None

    total_mebibytes_per_second = 0.0
    for value in numeric_rows[-1][2::3]:
        total_mebibytes_per_second += _float_or_none(value) or 0.0
    total_rate = total_mebibytes_per_second * 1024 * 1024
    return {
        "ioReadBytesPerSecond": None,
        "ioWriteBytesPerSecond": None,
        "ioTotalBytesPerSecond": round(total_rate, 2),
    }


def _windows_disk_io_rates() -> dict[str, float | None] | None:
    if not _is_windows() or psutil is None:
        return None
    counters = psutil.disk_io_counters()
    if counters is None:
        return None
    return _disk_io_rates_from_samples(
        now=time.monotonic(),
        read_bytes=int(counters.read_bytes),
        write_bytes=int(counters.write_bytes),
    )


def _uptime_seconds() -> float | None:
    uptime_path = Path("/proc/uptime")
    if uptime_path.exists():
        return round(float(uptime_path.read_text(encoding="utf-8").split()[0]), 1)

    output = _run_command(["sysctl", "-n", "kern.boottime"], timeout=5)
    match = re.search(r"sec = (\d+)", output)
    if match:
        return round(time.time() - int(match.group(1)), 1)

    if _is_windows() and psutil is not None:
        return round(time.time() - psutil.boot_time(), 1)
    return None


def _process_rss_bytes() -> int | None:
    if psutil is not None:
        try:
            return int(psutil.Process(os.getpid()).memory_info().rss)
        except (OSError, psutil.Error):
            pass
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if platform.system() == "Darwin":
        return int(usage.ru_maxrss)
    return int(usage.ru_maxrss * 1024)


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _run_command(command: list[str], *, timeout: int) -> str:
    try:
        return subprocess.check_output(
            command,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""
