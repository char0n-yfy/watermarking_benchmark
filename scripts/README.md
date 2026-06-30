# Service Startup Scripts

这些脚本用于一键拉起 WM Bench 的 Web UI、FastAPI 后端和本地 Worker。

启动脚本会在项目根目录创建或复用 `.venv`，并安装 `requirements.txt` 中的 Python 依赖。`requirements.txt` 汇总了 API、Worker 和评测算法运行依赖；SHARP/3D Viewpoint Re-rendering 的 CUDA 重型依赖单独放在 `requirements/sharp.txt`。

可选环境变量：

- `WM_BENCH_VENV`: 虚拟环境目录，默认 `.venv`。
- `WM_BENCH_DOTENV_PATH`: 指定要加载的 dotenv 文件，默认 `.env`；AutoDL 启动脚本会使用 `.env.autodl`。
- `WM_BENCH_INSTALL_PYTHON_DEPS`: 设为 `0` 时跳过依赖安装，但要求 `.venv` 已存在。
- `WM_BENCH_VENV_SYSTEM_SITE_PACKAGES`: 设为 `1` 时允许 `.venv` 读取系统 Python 包，AutoDL 默认开启以复用 CUDA/PyTorch。
- `WM_BENCH_INSTALL_SHARP_DEPS`: 设为 `1` 时安装 `requirements/sharp.txt`，默认开启；设为 `0` 可跳过 SHARP/3D 重型依赖。

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

需要跳过 SHARP/3D 攻击依赖时：

```bash
WM_BENCH_INSTALL_SHARP_DEPS=0 bash scripts/start-macos.sh
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

需要在 Windows 上跳过 SHARP/3D 攻击依赖时：

```powershell
$env:WM_BENCH_INSTALL_SHARP_DEPS = "0"
.\scripts\start-windows.ps1
```

## Linux / AutoDL

在 AutoDL 服务器的项目根目录执行这一条命令即可：

```bash
bash infra/autodl/start.sh
```

这条命令会创建 `.env.autodl`、创建或复用 `.venv`、安装 Python 依赖、在缺少 `screen` 或 Node.js/pnpm 时自动准备运行工具链、安装 Web 依赖、构建静态前端，并用 `screen` 启动 API 与 Worker。

兼容入口仍然可用：

```bash
bash scripts/start-autodl-linux.sh
```

该脚本只会转发到 `infra/autodl/start.sh`。

连接方式：

- 服务器本机: `http://127.0.0.1:6006`
- 健康检查: `http://127.0.0.1:6006/health`
- AutoDL 公网访问: 在 AutoDL 控制台把本机端口 `6006` 暴露为自定义服务或隧道，然后用控制台生成的公网 URL 访问。
- SSH 隧道访问:

```bash
ssh -L 6006:127.0.0.1:6006 root@<server-ip>
```

然后在自己的电脑打开 `http://127.0.0.1:6006`。

AutoDL 默认路径：

- 数据集: `<仓库根目录>/resources/datasets`
- 权重: `<仓库根目录>/resources/weights`
- 运行结果: `/root/autodl-tmp/wm-bench/runs`
- 日志: `/root/autodl-tmp/wm-bench/runs/logs`
- SQLite: `/root/autodl-fs/wm-bench/state/wmbench.sqlite`
- Python 虚拟环境: 项目根目录 `.venv`，默认允许读取 AutoDL 基础镜像中的系统包以复用 CUDA/PyTorch。

常用 AutoDL 配置写在 `.env.autodl` 中：

- `WM_BENCH_INSTALL_SHARP_DEPS=0`: 跳过 SHARP/3D 重型依赖。
- `WM_BENCH_AUTO_INSTALL_NODE=0`: 不自动安装 Node.js/pnpm，缺失时直接报错。
- `WM_BENCH_AUTO_INSTALL_SCREEN=0`: 不自动安装 `screen`，缺失时直接报错。
- `API_PORT=6006`: 修改服务端口。
- `WM_BENCH_DEVICE=cuda:0`: 修改 Worker 使用的设备。

## 服务关系

- Web UI 负责浏览器交互。
- FastAPI 提供配置、资源、运行队列、状态监控等接口。
- Worker 从队列中领取实验任务并实际执行。只启动 Web/API 时可以打开界面，但实验不会真正跑起来。

当前没有登录鉴权。不要把服务直接裸露到公网给不可信用户使用，优先使用 AutoDL 隧道、SSH 隧道、VPN 或带访问控制的反向代理。
