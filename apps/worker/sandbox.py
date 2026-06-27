from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SandboxSpec:
    image_ref: str
    input_dir: Path
    output_dir: Path
    weights_dir: Path | None = None
    gpu_device: str | None = None
    timeout_seconds: int = 3600
    env: dict[str, str] = field(default_factory=dict)


def docker_run_command(spec: SandboxSpec, command: list[str]) -> list[str]:
    if not command:
        raise ValueError("sandbox command must not be empty")

    docker_command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--mount",
        f"type=bind,source={spec.input_dir},target=/workspace/input,readonly",
        "--mount",
        f"type=bind,source={spec.output_dir},target=/workspace/output",
        "--workdir",
        "/workspace",
    ]

    if spec.weights_dir is not None:
        docker_command.extend(
            [
                "--mount",
                f"type=bind,source={spec.weights_dir},target=/workspace/weights,readonly",
            ]
        )

    if spec.gpu_device:
        docker_command.extend(["--gpus", f"device={spec.gpu_device}"])

    for key, value in sorted(spec.env.items()):
        docker_command.extend(["--env", f"{key}={value}"])

    docker_command.extend([spec.image_ref, *command])
    return docker_command
