#!/usr/bin/env python3
"""Canonical cross-platform lifecycle manager for WaterPrism services."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TIMEOUT = 90
SERVICE_NAMES = ("api", "worker", "web")


class WmBenchError(RuntimeError):
    pass


def parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not (key[0].isalpha() or key[0] == "_"):
            continue
        if not all(char.isalnum() or char == "_" for char in key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key] = value
    return values


def is_autodl_host() -> bool:
    return bool(
        os.environ.get("AUTODL_CONTAINER_ID")
        or os.environ.get("AUTODL_PROJECT_ID")
        or Path("/root/autodl-fs").exists()
        or Path("/root/autodl-tmp").exists()
    )


def resolve_profile(requested: str) -> str:
    if requested != "auto":
        return requested
    return "autodl" if platform.system() == "Linux" and is_autodl_host() else "local"


def default_env_file(profile: str) -> Path:
    if profile == "autodl":
        return PROJECT_ROOT / ".env.autodl"
    if profile == "production":
        return PROJECT_ROOT / ".env.production"
    return PROJECT_ROOT / ".env"


def ensure_profile_env(profile: str, path: Path) -> None:
    if path.exists():
        return
    example_name = ".env.autodl.example" if profile == "autodl" else ".env.production.example"
    example = PROJECT_ROOT / example_name
    if not example.is_file():
        raise WmBenchError(f"Missing environment template: {example}")
    shutil.copyfile(example, path)
    print(f"Created {path.relative_to(PROJECT_ROOT)} from {example_name}")


def path_from_env(raw: str, default: Path) -> Path:
    path = Path(raw or str(default)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def append_csv(values: str, additions: Iterable[str]) -> str:
    result = [item.strip() for item in values.split(",") if item.strip()]
    for item in additions:
        if item and item not in result:
            result.append(item)
    return ",".join(result)


@dataclass
class RuntimeConfig:
    profile: str
    env_file: Path
    env: dict[str, str]
    api_host: str
    api_port: int
    web_host: str
    web_port: int
    device: str
    venv_dir: Path
    resources_root: Path
    runs_root: Path
    log_dir: Path
    state_dir: Path
    mode: str

    @property
    def python(self) -> Path:
        if os.name == "nt":
            return self.venv_dir / "Scripts" / "python.exe"
        return self.venv_dir / "bin" / "python"

    @property
    def local_api_host(self) -> str:
        return "127.0.0.1" if self.api_host in {"0.0.0.0", "::"} else self.api_host

    @property
    def local_web_host(self) -> str:
        return "127.0.0.1" if self.web_host in {"0.0.0.0", "::"} else self.web_host

    @property
    def api_url(self) -> str:
        return f"http://{self.local_api_host}:{self.api_port}"

    @property
    def web_url(self) -> str:
        if self.mode == "static":
            return self.api_url
        return f"http://{self.local_web_host}:{self.web_port}"


def build_config(args: argparse.Namespace) -> RuntimeConfig:
    profile = resolve_profile(args.profile)
    env_file = Path(args.env_file).expanduser() if args.env_file else default_env_file(profile)
    if not env_file.is_absolute():
        env_file = (PROJECT_ROOT / env_file).resolve()
    if profile in {"autodl", "production"}:
        ensure_profile_env(profile, env_file)

    defaults = {
        "APP_ENV": "autodl" if profile == "autodl" else "production" if profile == "production" else "development",
        "API_HOST": "0.0.0.0" if profile in {"production", "autodl"} else "127.0.0.1",
        "API_PORT": "6006" if profile in {"production", "autodl"} else "8000",
        "WEB_HOST": "127.0.0.1",
        "WEB_PORT": "3000",
        "WM_BENCH_DEVICE": "cuda:0" if profile == "autodl" else "cpu",
        "WM_BENCH_WORKER_POLL_SECONDS": "2",
        "WM_BENCH_VENV": ".venv",
        "WM_BENCH_VENV_SYSTEM_SITE_PACKAGES": "1" if profile == "autodl" else "0",
        "WM_BENCH_INSTALL_PYTHON_DEPS": "1",
        "WM_BENCH_INSTALL_SHARP_DEPS": "1",
        "WM_BENCH_PREPARE_PERCEPTUAL_WEIGHTS": "1",
        "WM_BENCH_AUTO_INSTALL_NODE": "1" if profile == "autodl" else "0",
        "WM_BENCH_NODE_VERSION": "20",
        "WM_BENCH_PNPM_VERSION": "10.23.0",
        "WM_BENCH_RESOURCES_ROOT": "./resources",
        "WM_BENCH_RUNS_ROOT": "./runs",
        "NEXT_PUBLIC_API_BASE_URL": "" if profile in {"production", "autodl"} else "http://localhost:8000",
        "WM_BENCH_CORS_ORIGINS": (
            "http://localhost:3000,http://127.0.0.1:3000,"
            "http://localhost:6006,http://127.0.0.1:6006"
        ),
    }
    dotenv_values = parse_dotenv(env_file)
    env = defaults | dotenv_values | dict(os.environ)

    cli_overrides = {
        "API_HOST": args.api_host,
        "API_PORT": str(args.api_port) if args.api_port is not None else None,
        "WEB_HOST": args.web_host,
        "WEB_PORT": str(args.web_port) if args.web_port is not None else None,
        "WM_BENCH_DEVICE": args.device,
        "NEXT_PUBLIC_API_BASE_URL": args.api_url,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            env[key] = value
    if args.skip_sharp:
        env["WM_BENCH_INSTALL_SHARP_DEPS"] = "0"

    api_port = int(env["API_PORT"])
    web_port = int(env["WEB_PORT"])
    if not (1 <= api_port <= 65535 and 1 <= web_port <= 65535):
        raise WmBenchError("API and Web ports must be between 1 and 65535")
    if profile == "local" and api_port == web_port:
        raise WmBenchError("Local profile requires different API and Web ports")

    if args.api_url is None and profile == "local":
        api_port_overridden = args.api_port is not None or "API_PORT" in os.environ
        api_url_overridden = "NEXT_PUBLIC_API_BASE_URL" in os.environ
        if api_port_overridden and not api_url_overridden:
            env["NEXT_PUBLIC_API_BASE_URL"] = f"http://localhost:{api_port}"

    origins = [f"http://localhost:{web_port}", f"http://127.0.0.1:{web_port}"]
    if env["WEB_HOST"] not in {"0.0.0.0", "::", "127.0.0.1", "localhost"}:
        origins.append(f"http://{env['WEB_HOST']}:{web_port}")
    origins.extend(args.cors_origin or [])
    env["WM_BENCH_CORS_ORIGINS"] = append_csv(env.get("WM_BENCH_CORS_ORIGINS", ""), origins)
    env["NEXT_PUBLIC_API_PORT"] = str(api_port)
    env["WM_BENCH_DOTENV_PATH"] = str(env_file)

    resources_root = path_from_env(env["WM_BENCH_RESOURCES_ROOT"], PROJECT_ROOT / "resources")
    runs_root = path_from_env(env["WM_BENCH_RUNS_ROOT"], PROJECT_ROOT / "runs")
    venv_dir = path_from_env(env["WM_BENCH_VENV"], PROJECT_ROOT / ".venv")
    log_dir = path_from_env(env.get("WM_BENCH_LOG_DIR", ""), runs_root / "logs")
    state_dir = runs_root / "services" / profile

    env.update(
        {
            "API_HOST": env["API_HOST"],
            "API_PORT": str(api_port),
            "WEB_HOST": env["WEB_HOST"],
            "WEB_PORT": str(web_port),
            "WM_BENCH_DEVICE": env["WM_BENCH_DEVICE"],
            "WM_BENCH_RESOURCES_ROOT": str(resources_root),
            "WM_BENCH_RUNS_ROOT": str(runs_root),
            "WM_BENCH_VENV": str(venv_dir),
            "WM_BENCH_LOG_DIR": str(log_dir),
        }
    )
    return RuntimeConfig(
        profile=profile,
        env_file=env_file,
        env=env,
        api_host=env["API_HOST"],
        api_port=api_port,
        web_host=env["WEB_HOST"],
        web_port=web_port,
        device=env["WM_BENCH_DEVICE"],
        venv_dir=venv_dir,
        resources_root=resources_root,
        runs_root=runs_root,
        log_dir=log_dir,
        state_dir=state_dir,
        mode="dev" if profile == "local" else "static",
    )


def ensure_dirs(config: RuntimeConfig) -> None:
    for path in (
        config.resources_root / "datasets",
        config.resources_root / "weights",
        config.runs_root,
        config.log_dir,
        config.state_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)


def run(command: list[str], config: RuntimeConfig, *, env: dict[str, str] | None = None) -> None:
    print("+", subprocess.list2cmdline(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env or config.env, check=False)
    if completed.returncode:
        raise WmBenchError(f"Command failed with exit code {completed.returncode}: {command[0]}")


def hash_paths(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(PROJECT_ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def python_fingerprint(config: RuntimeConfig) -> str:
    paths = [PROJECT_ROOT / "requirements.txt"]
    paths.extend((PROJECT_ROOT / "requirements").glob("*.txt"))
    paths.extend((PROJECT_ROOT / "apps").glob("*/requirements.txt"))
    digest = hash_paths(paths)
    return f"{sys.version_info.major}.{sys.version_info.minor}:{config.env.get('WM_BENCH_INSTALL_SHARP_DEPS')}:{digest}"


def web_install_fingerprint() -> str:
    return hash_paths(
        [
            PROJECT_ROOT / "package.json",
            PROJECT_ROOT / "pnpm-lock.yaml",
            PROJECT_ROOT / "pnpm-workspace.yaml",
            PROJECT_ROOT / "apps" / "web" / "package.json",
        ]
    )


def web_build_fingerprint(config: RuntimeConfig) -> str:
    roots = [
        PROJECT_ROOT / "apps" / "web" / name
        for name in ("app", "components", "data", "lib", "public", "styles")
    ]
    paths: list[Path] = [PROJECT_ROOT / "apps" / "web" / "next.config.mjs"]
    for root in roots:
        if root.exists():
            paths.extend(path for path in root.rglob("*") if path.is_file())
    return f"{web_install_fingerprint()}:{config.env.get('NEXT_PUBLIC_API_BASE_URL', '')}:{hash_paths(paths)}"


def bootstrap_state_path(config: RuntimeConfig) -> Path:
    return config.state_dir / "bootstrap.json"


def read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def node_major(config: RuntimeConfig) -> int:
    node = shutil.which("node", path=config.env.get("PATH"))
    if not node:
        return 0
    completed = subprocess.run(
        [node, "-p", 'process.versions.node.split(".")[0]'],
        cwd=PROJECT_ROOT,
        env=config.env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    try:
        return int(completed.stdout.strip())
    except ValueError:
        return 0


def ensure_node(config: RuntimeConfig) -> None:
    required = int(config.env.get("WM_BENCH_NODE_VERSION", "20"))
    if node_major(config) >= required:
        return
    if config.env.get("WM_BENCH_AUTO_INSTALL_NODE", "0") == "1" and shutil.which("conda"):
        print(f"Installing Node.js {required}.x with conda...")
        run(
            [
                shutil.which("conda") or "conda",
                "install",
                "-y",
                "--freeze-installed",
                "-c",
                "conda-forge",
                f"nodejs>={required},<{required + 1}",
            ],
            config,
        )
        if node_major(config) >= required:
            return
    raise WmBenchError(f"Node.js {required}+ is required. Install it and rerun bootstrap.")


def pnpm_command(config: RuntimeConfig) -> list[str]:
    pnpm = shutil.which("pnpm", path=config.env.get("PATH"))
    if pnpm:
        return [pnpm]
    corepack = shutil.which("corepack", path=config.env.get("PATH"))
    if corepack:
        probe = subprocess.run(
            [corepack, "pnpm", "--version"],
            cwd=PROJECT_ROOT,
            env=config.env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if probe.returncode == 0:
            return [corepack, "pnpm"]
    npm = shutil.which("npm", path=config.env.get("PATH"))
    if npm and config.env.get("WM_BENCH_AUTO_INSTALL_NODE", "0") == "1":
        version = config.env.get("WM_BENCH_PNPM_VERSION", "10.23.0")
        run([npm, "install", "-g", f"pnpm@{version}"], config)
        pnpm = shutil.which("pnpm", path=config.env.get("PATH"))
        if pnpm:
            return [pnpm]
    raise WmBenchError("pnpm is required. Enable corepack or install pnpm and rerun bootstrap.")


def prepare_python(config: RuntimeConfig, state: dict[str, object], *, force: bool) -> None:
    if config.env.get("WM_BENCH_INSTALL_PYTHON_DEPS", "1") == "0":
        if not config.python.is_file():
            raise WmBenchError(
                f"Missing Python environment: {config.python}. "
                "Unset WM_BENCH_INSTALL_PYTHON_DEPS or create it manually."
            )
        print(f"Python dependency install skipped: {config.python}")
        return

    fingerprint = python_fingerprint(config)
    if not force and config.python.is_file() and state.get("pythonFingerprint") == fingerprint:
        print(f"Python environment is current: {config.python}")
        return
    if not config.python.is_file():
        command = [sys.executable, "-m", "venv"]
        if config.env.get("WM_BENCH_VENV_SYSTEM_SITE_PACKAGES", "0") != "0":
            command.append("--system-site-packages")
        command.append(str(config.venv_dir))
        run(command, config)
    run([str(config.python), "-m", "pip", "install", "--upgrade", "pip", "setuptools<82", "wheel"], config)
    requirements = ["-r", "requirements.txt"]
    if config.env.get("WM_BENCH_INSTALL_SHARP_DEPS", "1") != "0":
        requirements.extend(["-r", "requirements/sharp.txt"])
    run([str(config.python), "-m", "pip", "install", *requirements], config)
    state["pythonFingerprint"] = fingerprint


def prepare_perceptual_weights(config: RuntimeConfig, state: dict[str, object], *, force: bool) -> None:
    if config.profile != "autodl" or config.env.get("WM_BENCH_PREPARE_PERCEPTUAL_WEIGHTS", "1") == "0":
        return
    script = PROJECT_ROOT / "scripts" / "prepare_perceptual_weights.py"
    fingerprint = hash_paths([script])
    if not force and state.get("perceptualWeightsFingerprint") == fingerprint:
        print("Perceptual metric weights are prepared.")
        return
    run([str(config.python), str(script.relative_to(PROJECT_ROOT))], config)
    state["perceptualWeightsFingerprint"] = fingerprint


def _windows_lhm_candidates(config: RuntimeConfig) -> list[Path]:
    candidates: list[Path] = []
    configured = config.env.get("WM_BENCH_LHM_PATH", "").strip()
    if configured:
        path = Path(configured).expanduser()
        candidates.append(path if path.is_absolute() else PROJECT_ROOT / path)

    candidates.extend(
        [
            PROJECT_ROOT / "LibreHardwareMonitor" / "LibreHardwareMonitor.exe",
            PROJECT_ROOT.parent / "LibreHardwareMonitor" / "LibreHardwareMonitor.exe",
        ]
    )
    for key, suffix in (
        ("ProgramFiles", ("LibreHardwareMonitor", "LibreHardwareMonitor.exe")),
        ("ProgramFiles(x86)", ("LibreHardwareMonitor", "LibreHardwareMonitor.exe")),
        ("LOCALAPPDATA", ("Programs", "LibreHardwareMonitor", "LibreHardwareMonitor.exe")),
    ):
        root = config.env.get(key, "").strip()
        if root:
            candidates.append(Path(root).joinpath(*suffix))

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def maybe_start_windows_hardware_monitor(config: RuntimeConfig) -> None:
    if platform.system() != "Windows" or config.env.get("WM_BENCH_SKIP_LHM", "0") == "1":
        return

    executable = next((path for path in _windows_lhm_candidates(config) if path.is_file()), None)
    if executable is None:
        return

    tasklist = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq LibreHardwareMonitor.exe", "/NH"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if "librehardwaremonitor.exe" in tasklist.stdout.lower():
        return

    startupinfo = subprocess.STARTUPINFO()  # type: ignore[attr-defined]
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # type: ignore[attr-defined]
    startupinfo.wShowWindow = 2  # SW_SHOWMINIMIZED
    subprocess.Popen(
        [str(executable)],
        cwd=executable.parent,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        startupinfo=startupinfo,
        creationflags=(
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        ),
        close_fds=True,
    )
    print(f"LibreHardwareMonitor started: {executable}")


def prepare_web(config: RuntimeConfig, state: dict[str, object], *, force: bool, build: bool) -> None:
    ensure_node(config)
    pnpm = pnpm_command(config)
    install_fingerprint = web_install_fingerprint()
    node_modules = PROJECT_ROOT / "node_modules"
    if force or not node_modules.is_dir() or state.get("webInstallFingerprint") != install_fingerprint:
        install_env = dict(config.env)
        install_env["CI"] = "true"
        run([*pnpm, "install", "--frozen-lockfile"], config, env=install_env)
        state["webInstallFingerprint"] = install_fingerprint
    else:
        print("Web dependencies are current.")

    if not build:
        return
    build_fingerprint = web_build_fingerprint(config)
    web_out = PROJECT_ROOT / "apps" / "web" / "out" / "index.html"
    if force or not web_out.is_file() or state.get("webBuildFingerprint") != build_fingerprint:
        run([*pnpm, "--filter", "@wm-bench/web", "build"], config)
        if not web_out.is_file():
            raise WmBenchError("Web build completed without producing apps/web/out/index.html")
        state["webBuildFingerprint"] = build_fingerprint
    else:
        print("Static Web build is current.")


def bootstrap(config: RuntimeConfig, *, force: bool = False) -> None:
    ensure_dirs(config)
    state_path = bootstrap_state_path(config)
    state = read_json(state_path)
    prepare_python(config, state, force=force)
    prepare_perceptual_weights(config, state, force=force)
    prepare_web(config, state, force=force, build=config.mode == "static")
    state["profile"] = config.profile
    state["updatedAt"] = time.time()
    write_json(state_path, state)
    print(f"Bootstrap complete for profile {config.profile}.")


def command_line(pid: int) -> str:
    proc_cmdline = Path(f"/proc/{pid}/cmdline")
    if proc_cmdline.is_file():
        try:
            return proc_cmdline.read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            return ""
    if os.name == "nt":
        script = f"(Get-CimInstance Win32_Process -Filter 'ProcessId = {pid}').CommandLine"
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return completed.stdout.strip()
    completed = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip()


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def service_meta_path(config: RuntimeConfig, name: str) -> Path:
    return config.state_dir / f"{name}.json"


def service_metadata(config: RuntimeConfig, name: str) -> dict[str, object]:
    return read_json(service_meta_path(config, name))


def service_is_active(config: RuntimeConfig, name: str) -> bool:
    meta = service_metadata(config, name)
    try:
        pid = int(meta.get("pid", 0))
    except (TypeError, ValueError):
        return False
    if not pid_alive(pid):
        return False
    marker = str(meta.get("marker", ""))
    line = command_line(pid)
    return bool(marker and marker in line)


def spawn_service(config: RuntimeConfig, name: str, command: list[str], marker: str) -> subprocess.Popen[bytes]:
    log_path = config.log_dir / f"{name}.log"
    log_handle = log_path.open("ab", buffering=0)
    kwargs: dict[str, object] = {
        "cwd": PROJECT_ROOT,
        "env": config.env,
        "stdin": subprocess.DEVNULL,
        "stdout": log_handle,
        "stderr": subprocess.STDOUT,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        kwargs["start_new_session"] = True
    try:
        process = subprocess.Popen(command, **kwargs)  # type: ignore[arg-type]
    finally:
        log_handle.close()
    write_json(
        service_meta_path(config, name),
        {
            "pid": process.pid,
            "name": name,
            "profile": config.profile,
            "marker": marker,
            "command": command,
            "log": str(log_path),
            "startedAt": time.time(),
            "projectRoot": str(PROJECT_ROOT),
        },
    )
    return process


def stop_service(config: RuntimeConfig, name: str, *, quiet: bool = False) -> bool:
    meta_path = service_meta_path(config, name)
    meta = read_json(meta_path)
    try:
        pid = int(meta.get("pid", 0))
    except (TypeError, ValueError):
        pid = 0
    if not pid or not pid_alive(pid):
        meta_path.unlink(missing_ok=True)
        return False
    marker = str(meta.get("marker", ""))
    line = command_line(pid)
    if not marker or marker not in line:
        raise WmBenchError(
            f"Refusing to stop PID {pid} for {name}: process identity no longer matches {marker!r}. "
            f"Remove stale metadata manually: {meta_path}"
        )
    if not quiet:
        print(f"Stopping {name} (PID {pid})...")
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for _ in range(40):
        if not pid_alive(pid):
            break
        time.sleep(0.25)
    if pid_alive(pid):
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid), "/T"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    meta_path.unlink(missing_ok=True)
    return True


def stop(config: RuntimeConfig, *, quiet: bool = False) -> None:
    stopped = False
    for name in reversed(SERVICE_NAMES):
        stopped = stop_service(config, name, quiet=quiet) or stopped
    if not quiet:
        print("WM Bench services stopped." if stopped else "WM Bench services are not running.")


def port_available(host: str, port: int) -> bool:
    bind_host = "127.0.0.1" if host in {"0.0.0.0", "::", "localhost"} else host
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((bind_host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def get_json(url: str, timeout: float = 2.0) -> dict[str, object] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
            return value if isinstance(value, dict) else None
    except (OSError, ValueError, urllib.error.URLError):
        return None


def url_reachable(url: str, timeout: float = 2.0) -> bool:
    try:
        request = urllib.request.Request(url, headers={"Accept": "text/html"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= response.status < 500
    except (OSError, urllib.error.URLError):
        return False


def tail_log(config: RuntimeConfig, name: str, lines: int = 60) -> str:
    path = config.log_dir / f"{name}.log"
    if not path.is_file():
        return f"Log not found: {path}"
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def wait_for_start(config: RuntimeConfig, started: list[str], timeout: int) -> None:
    deadline = time.monotonic() + timeout
    api_payload: dict[str, object] | None = None
    while time.monotonic() < deadline:
        for name in started:
            if not service_is_active(config, name):
                raise WmBenchError(f"{name} exited during startup.\n{tail_log(config, name)}")
        api_payload = get_json(f"{config.api_url}/health")
        if api_payload:
            roots_match = Path(str(api_payload.get("runs_root", ""))).resolve() == config.runs_root
            if (
                api_payload.get("environment") == config.env["APP_ENV"]
                and str(api_payload.get("configured_api_port")) == str(config.api_port)
                and roots_match
            ):
                break
        time.sleep(1)
    else:
        raise WmBenchError(f"API did not become healthy at {config.api_url}/health\n{tail_log(config, 'api')}")

    web_deadline = time.monotonic() + timeout
    while time.monotonic() < web_deadline:
        if url_reachable(config.web_url):
            break
        if config.mode == "dev" and not service_is_active(config, "web"):
            raise WmBenchError(f"Web exited during startup.\n{tail_log(config, 'web')}")
        time.sleep(1)
    else:
        raise WmBenchError(f"Web did not become reachable at {config.web_url}\n{tail_log(config, 'web')}")

    worker_deadline = time.monotonic() + min(timeout, 30)
    while time.monotonic() < worker_deadline:
        runtime = get_json(f"{config.api_url}/system/runtime") or {}
        if int(runtime.get("knownWorkerCount", 0) or 0) > 0:
            return
        if not service_is_active(config, "worker"):
            raise WmBenchError(f"Worker exited during startup.\n{tail_log(config, 'worker')}")
        time.sleep(1)
    raise WmBenchError(f"Worker did not publish a fresh heartbeat.\n{tail_log(config, 'worker')}")


def start(config: RuntimeConfig, *, no_bootstrap: bool, timeout: int) -> None:
    ensure_dirs(config)
    active = [name for name in SERVICE_NAMES if service_is_active(config, name)]
    if active:
        raise WmBenchError(f"Services already running: {', '.join(active)}. Use restart or down first.")
    if not no_bootstrap:
        bootstrap(config)
    elif not config.python.is_file():
        raise WmBenchError(f"Missing Python environment: {config.python}. Run bootstrap first.")

    maybe_start_windows_hardware_monitor(config)

    ports = [("API", config.api_host, config.api_port)]
    if config.mode == "dev":
        ports.append(("Web", config.web_host, config.web_port))
    for label, host, port in ports:
        if not port_available(host, port):
            raise WmBenchError(f"{label} port {port} is already occupied; refusing to start.")

    started: list[str] = []
    try:
        spawn_service(
            config,
            "api",
            [
                str(config.python),
                "-m",
                "uvicorn",
                "app.main:app",
                "--app-dir",
                "apps/api",
                "--host",
                config.api_host,
                "--port",
                str(config.api_port),
            ],
            "uvicorn app.main:app",
        )
        started.append("api")
        spawn_service(
            config,
            "worker",
            [
                str(config.python),
                "apps/worker/local_worker.py",
                "--poll-seconds",
                config.env["WM_BENCH_WORKER_POLL_SECONDS"],
            ],
            "apps/worker/local_worker.py",
        )
        started.append("worker")
        if config.mode == "dev":
            pnpm = pnpm_command(config)
            spawn_service(
                config,
                "web",
                [
                    *pnpm,
                    "--filter",
                    "@wm-bench/web",
                    "dev",
                    "--hostname",
                    config.web_host,
                    "--port",
                    str(config.web_port),
                ],
                "@wm-bench/web dev",
            )
            started.append("web")
        wait_for_start(config, started, timeout)
    except Exception:
        for name in reversed(started):
            try:
                stop_service(config, name, quiet=True)
            except WmBenchError:
                pass
        raise

    print("WM Bench services started.")
    print(f"Profile:    {config.profile}")
    print(f"Web UI:     {config.web_url}")
    print(f"API health: {config.api_url}/health")
    print(f"Logs:       {config.log_dir}")
    print(f"Stop:       {sys.executable} scripts/wmbench.py --profile {config.profile} down")


def status(config: RuntimeConfig) -> int:
    print(f"Profile: {config.profile}")
    all_active = True
    expected = ("api", "worker") if config.mode == "static" else SERVICE_NAMES
    for name in expected:
        meta = service_metadata(config, name)
        pid = meta.get("pid", "-")
        active = service_is_active(config, name)
        all_active = all_active and active
        print(f"  {name:<6} {'running' if active else 'stopped':<8} pid={pid}")
    health = get_json(f"{config.api_url}/health")
    print(f"API:     {'healthy' if health else 'unreachable'} ({config.api_url}/health)")
    web_reachable = url_reachable(config.web_url)
    print(f"Web:     {'reachable' if web_reachable else 'unreachable'} ({config.web_url})")
    return 0 if all_active and health and web_reachable else 1


def show_logs(config: RuntimeConfig, *, lines: int, follow: bool) -> None:
    names = ("api", "worker") if config.mode == "static" else SERVICE_NAMES
    if follow:
        if os.name == "nt":
            raise WmBenchError("--follow is not supported on Windows; open the files shown below instead.")
        paths = [str(config.log_dir / f"{name}.log") for name in names]
        os.execvp("tail", ["tail", "-n", str(lines), "-f", *paths])
    for name in names:
        path = config.log_dir / f"{name}.log"
        print(f"==> {path}")
        print(tail_log(config, name, lines))
        print()


def check(config: RuntimeConfig) -> int:
    ensure_dirs(config)
    failures: list[str] = []
    warnings: list[str] = []
    print(f"WM Bench deployment check ({config.profile})")
    if config.python.is_file():
        print(f"[PASS] Python environment: {config.python}")
        completed = subprocess.run(
            [str(config.python), "-m", "pip", "check"],
            cwd=PROJECT_ROOT,
            env=config.env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if completed.returncode:
            failures.append(f"Python dependencies: {completed.stdout.strip()}")
        else:
            print("[PASS] Python dependencies")
        readiness = subprocess.run(
            [str(config.python), "scripts/check-deploy-readiness.py", "--json"],
            cwd=PROJECT_ROOT,
            env=config.env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if readiness.returncode:
            detail = readiness.stdout.strip() or readiness.stderr.strip()
            failures.append(f"Application readiness failed: {detail}")
        else:
            try:
                readiness_payload = json.loads(readiness.stdout)
                readiness_status = str(readiness_payload.get("status", "unknown"))
                if readiness_status == "ready":
                    print("[PASS] Application readiness: ready")
                else:
                    warnings.append(f"Application readiness: {readiness_status}")
            except ValueError:
                warnings.append("Application readiness returned non-JSON output")
        if config.device.startswith("cuda"):
            cuda_probe = subprocess.run(
                [
                    str(config.python),
                    "-c",
                    "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)",
                ],
                cwd=PROJECT_ROOT,
                env=config.env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            if cuda_probe.returncode:
                failures.append(f"Configured device {config.device} but PyTorch CUDA is unavailable")
            else:
                print(f"[PASS] PyTorch device: {config.device}")
    else:
        failures.append(f"Missing Python environment: {config.python}")
    required_node = int(config.env.get("WM_BENCH_NODE_VERSION", "20"))
    found_node = node_major(config)
    if found_node >= required_node:
        print(f"[PASS] Node.js major version: {found_node}")
        try:
            pnpm = pnpm_command(config)
            print(f"[PASS] pnpm command: {subprocess.list2cmdline(pnpm)}")
        except WmBenchError as exc:
            failures.append(str(exc))
    else:
        failures.append(f"Node.js {required_node}+ is required; found {found_node or 'none'}")
    if config.mode == "static":
        output = PROJECT_ROOT / "apps" / "web" / "out" / "index.html"
        state = read_json(bootstrap_state_path(config))
        if not output.is_file():
            failures.append("Missing static Web build: apps/web/out/index.html")
        elif state.get("webBuildFingerprint") != web_build_fingerprint(config):
            warnings.append("Static Web build is stale or predates the unified bootstrap state; run bootstrap")
        else:
            print("[PASS] Static Web build is current")
    for label, host, port in [("API", config.api_host, config.api_port)] + (
        [("Web", config.web_host, config.web_port)] if config.mode == "dev" else []
    ):
        if port_available(host, port):
            print(f"[PASS] {label} port available: {port}")
        elif service_is_active(config, label.lower()):
            warnings.append(f"{label} port {port} is occupied by the running WM Bench service")
        else:
            failures.append(f"{label} port {port} is occupied by another process")
    for warning in warnings:
        print(f"[WARN] {warning}")
    for failure in failures:
        print(f"[FAIL] {failure}")
    if failures:
        print("Result: not ready")
        return 1
    print("Result: ready with warnings" if warnings else "Result: ready")
    return 0


def tunnel(config: RuntimeConfig) -> None:
    public_url = config.env.get("WM_BENCH_PUBLIC_URL", "").rstrip("/")
    if not public_url:
        public_url = config.env.get(f"AutoDLService{config.api_port}URL", "").rstrip("/")
    if not public_url and config.api_port == 6006:
        public_url = config.env.get("AutoDLServiceURL", "").rstrip("/")
    print(f"Server-local URL: {config.api_url}")
    if public_url:
        print(f"Configured public URL: {public_url}")
    print("SSH tunnel from your computer:")
    print(f"  ssh -L {config.api_port}:127.0.0.1:{config.api_port} root@<server-ip>")
    print(f"  open http://127.0.0.1:{config.api_port}")
    print("Security: the service has no login layer; do not expose it to untrusted public users.")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WaterPrism cross-platform service manager")
    parser.add_argument("--profile", choices=("auto", "local", "production", "autodl"), default="auto")
    parser.add_argument("--env-file", help="dotenv path; defaults to .env or .env.autodl")
    parser.add_argument("--api-host")
    parser.add_argument("--api-port", type=int)
    parser.add_argument("--web-host")
    parser.add_argument("--web-port", type=int)
    parser.add_argument("--device")
    parser.add_argument("--api-url", help="browser-visible API base URL for local development")
    parser.add_argument("--cors-origin", action="append", help="additional allowed browser origin")
    parser.add_argument("--skip-sharp", action="store_true", help="skip optional SHARP/3D dependencies")
    subparsers = parser.add_subparsers(dest="command", required=True)
    bootstrap_parser = subparsers.add_parser(
        "bootstrap", help="prepare Python/Node dependencies and production Web build"
    )
    bootstrap_parser.add_argument("--force", action="store_true")
    up_parser = subparsers.add_parser("up", help="prepare and start API, Worker, and Web")
    up_parser.add_argument("--no-bootstrap", action="store_true")
    up_parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    restart_parser = subparsers.add_parser("restart", help="stop and start services")
    restart_parser.add_argument("--no-bootstrap", action="store_true")
    restart_parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    subparsers.add_parser("down", aliases=["stop"], help="stop services owned by this profile")
    subparsers.add_parser("status", help="show process and HTTP status")
    logs_parser = subparsers.add_parser("logs", help="show service logs")
    logs_parser.add_argument("--lines", type=int, default=80)
    logs_parser.add_argument("--follow", action="store_true")
    subparsers.add_parser("check", help="check deployment prerequisites without starting services")
    subparsers.add_parser("tunnel", help="show remote access instructions")
    return parser


def main() -> int:
    args = create_parser().parse_args()
    try:
        config = build_config(args)
        command = args.command
        if command == "bootstrap":
            bootstrap(config, force=args.force)
        elif command == "up":
            start(config, no_bootstrap=args.no_bootstrap, timeout=args.timeout)
        elif command == "restart":
            if not args.no_bootstrap:
                bootstrap(config)
            stop(config)
            start(config, no_bootstrap=True, timeout=args.timeout)
        elif command in {"down", "stop"}:
            stop(config)
        elif command == "status":
            return status(config)
        elif command == "logs":
            show_logs(config, lines=args.lines, follow=args.follow)
        elif command == "check":
            return check(config)
        elif command == "tunnel":
            tunnel(config)
        return 0
    except (WmBenchError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
