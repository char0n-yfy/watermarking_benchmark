from __future__ import annotations

import os
import platform
import re
import shutil

try:
    import resource
except ModuleNotFoundError:
    resource = None  # Windows has no resource module
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_DISK_IO_SAMPLE: tuple[float, int, int] | None = None


def collect_system_metrics(*, data_root: Path, device: str) -> dict[str, Any]:
    disk = shutil.disk_usage(data_root)
    memory = _memory_metrics()
    cpu_percent = _cpu_usage_percent()
    uptime_seconds = _uptime_seconds()
    disk_io = _disk_io_rates()

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
    return None


def _memory_metrics() -> dict[str, int | float | None]:
    linux_memory = _linux_memory_metrics()
    if linux_memory is not None:
        return linux_memory
    darwin_memory = _darwin_memory_metrics()
    if darwin_memory is not None:
        return darwin_memory
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


def _gpu_metrics() -> dict[str, Any]:
    output = _run_command(
        [
            "nvidia-smi",
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
    return {
        "ioReadBytesPerSecond": None,
        "ioWriteBytesPerSecond": None,
        "ioTotalBytesPerSecond": None,
    }


def _linux_disk_io_rates() -> dict[str, float | None] | None:
    global _DISK_IO_SAMPLE

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

    now = time.monotonic()
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


def _uptime_seconds() -> float | None:
    uptime_path = Path("/proc/uptime")
    if uptime_path.exists():
        return round(float(uptime_path.read_text(encoding="utf-8").split()[0]), 1)

    output = _run_command(["sysctl", "-n", "kern.boottime"], timeout=5)
    match = re.search(r"sec = (\d+)", output)
    if match:
        return round(time.time() - int(match.group(1)), 1)
    return None


def _process_rss_bytes() -> int | None:
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
