# Service Startup Scripts

这些脚本用于一键拉起 WM Bench 的 Web UI、FastAPI 后端和本地 Worker。

## 首次配置（OSS 数据集下载）

**默认无需 AccessKey。** 项目已内置团队 OSS 地址（`watermarking-benchmark` / `wmbench` 前缀），开启公开读后，克隆仓库、启动脚本即可在资源页下载。

管理员只需在阿里云 OSS 控制台做一次：**将 `wmbench/datasets/` 设为公共读**（或整个 bucket 公共读，仅建议团队内网使用）。

可选：复制 `.env.example` 为 `.env` 以覆盖默认项（`.env` 已在 `.gitignore` 中，不会提交）：

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
copy .env.example .env
```

私有 bucket 时：在 `.env` 中设置 `WM_BENCH_OSS_PUBLIC_READ=false` 并填写 `WM_BENCH_OSS_ACCESS_KEY` / `WM_BENCH_OSS_SECRET_KEY`。

验证：打开 `http://127.0.0.1:8000/resources/storage/status`，应看到 `"enabled": true`, `"mode": "public-read"`。

## macOS

在项目根目录执行：

```bash
bash scripts/start-macos.sh
```

连接方式：

- Web UI: `http://127.0.0.1:3000`
- API 健康检查: `http://127.0.0.1:8000/health`
- 日志: `runs/local/logs/`

默认使用本机 CPU。需要指定端口或设备时：

```bash
API_PORT=8001 WEB_PORT=3001 WM_BENCH_DEVICE=mps bash scripts/start-macos.sh
```

## Windows

在 PowerShell 中进入项目根目录：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\start-windows.ps1
```

连接方式：

- Web UI: `http://127.0.0.1:3000`
- API 健康检查: `http://127.0.0.1:8000/health`
- 日志: `runs\local\logs\`

指定端口或设备：

```powershell
.\scripts\start-windows.ps1 -ApiPort 8001 -WebPort 3001 -Device cpu
```

## Linux / AutoDL

在 AutoDL 服务器的项目根目录执行：

```bash
bash scripts/start-autodl-linux.sh
```

连接方式：

- 服务器本机: `http://127.0.0.1:6006`
- AutoDL 公网访问: 在 AutoDL 控制台把本机端口 `6006` 暴露为自定义服务或隧道，然后用控制台生成的公网 URL 访问。
- SSH 隧道访问:

```bash
ssh -L 6006:127.0.0.1:6006 root@<server-ip>
```

然后在自己的电脑打开 `http://127.0.0.1:6006`。

AutoDL 默认路径：

- 数据集: `/root/autodl-fs/wm-bench/resources/datasets`
- 权重: `/root/autodl-fs/wm-bench/resources/weights`
- 运行结果: `/root/autodl-tmp/wm-bench/runs`
- SQLite: `/root/autodl-fs/wm-bench/state/wmbench.sqlite`
- Python 虚拟环境: 项目根目录 `.venv`，默认允许读取 AutoDL 基础镜像中的系统包以复用 CUDA/PyTorch。

## 服务关系

- Web UI 负责浏览器交互。
- FastAPI 提供配置、资源、运行队列、状态监控等接口。
- Worker 从队列中领取实验任务并实际执行。只启动 Web/API 时可以打开界面，但实验不会真正跑起来。

当前没有登录鉴权。不要把服务直接裸露到公网给不可信用户使用，优先使用 AutoDL 隧道、SSH 隧道、VPN 或带访问控制的反向代理。
