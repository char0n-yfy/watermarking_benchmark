# Watermarking Benchmark

图像水印鲁棒性 Benchmark Web 系统。当前主运行形态是单机部署：FastAPI 提供资源目录、实验配置、运行队列和静态前端服务，本地 Worker 轮询队列并执行水印嵌入、攻击、检测和质量评估任务。

正式实验推荐运行在 Linux / AutoDL。macOS 和 Windows 主要用于界面调试、资源管理、CPU 子集实验，部分 CUDA、3D/SHARP 或重型模型依赖可能不可用。

## 目录结构

```text
apps/
  api/       FastAPI 后端服务
  web/       Next.js 前端，生产构建输出到 apps/web/out
  worker/    本地实验执行 Worker
evaluator/   水印算法、攻击算法和评估逻辑
infra/       AutoDL 部署与运行脚本
scripts/     跨平台启动、依赖初始化、部署检查和维护脚本
docs/        评分协议和开发规范
resources/   数据集、权重、资源元数据入口
requirements/ 可选或分层 Python 依赖
runs/        本地运行结果、日志和实验状态，默认不进入 Git
```

源码仓库只保存代码、配置模板、脚本和小型元数据。真实数据集、模型权重、运行结果和缓存属于外部资源或本机运行产物，不应提交到 Git。

## 一键启动与关闭

| 平台 | 启动服务 | 关闭服务 |
| --- | --- | --- |
| macOS | `bash scripts/start-macos.sh` | `bash scripts/stop-macos.sh` |
| Windows PowerShell | `.\scripts\start-windows.ps1` | `.\scripts\stop-windows.ps1` |
| Linux / AutoDL | `bash scripts/deploy-autodl-linux.sh` | `bash scripts/deploy-autodl-linux.sh stop` |

Windows 首次运行脚本时，如遇到执行策略限制，可以先在当前 PowerShell 会话执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

启动后打开脚本输出的 Web URL。AutoDL 默认使用 `6006`，FastAPI 会在同一端口托管 `apps/web/out` 静态前端，浏览器和 API 保持同源访问。关闭命令只停止当前项目启动的本地服务，不会删除数据集、权重或历史实验结果。

## 平台说明

macOS 本地开发默认端口：

- Web UI: `http://127.0.0.1:3000`
- API: `http://127.0.0.1:8000`
- 默认设备: `cpu`

可通过环境变量覆盖：

```bash
API_PORT=8001 WEB_PORT=3001 WM_BENCH_DEVICE=mps bash scripts/start-macos.sh
API_PORT=8001 WEB_PORT=3001 bash scripts/stop-macos.sh
```

Windows 本地开发默认端口同 macOS。可通过参数覆盖：

```powershell
.\scripts\start-windows.ps1 -ApiPort 8001 -WebPort 3001 -Device cpu
.\scripts\stop-windows.ps1 -ApiPort 8001 -WebPort 3001
```

Windows 上如需更准确的 CPU 功耗读数，可安装 LibreHardwareMonitor，并在 `.env` 中设置 `WM_BENCH_LHM_PATH`。不需要该能力时可设置 `WM_BENCH_SKIP_LHM=1`。

AutoDL 常用运维命令：

```bash
bash scripts/deploy-autodl-linux.sh status
bash scripts/deploy-autodl-linux.sh logs
bash scripts/deploy-autodl-linux.sh tunnel
bash scripts/deploy-autodl-linux.sh restart
bash scripts/deploy-autodl-linux.sh stop
```

AutoDL 默认只暴露一个端口 `6006`。可以在 AutoDL 控制台创建自定义服务，也可以从本机建立 SSH 隧道：

```bash
ssh -L 6006:127.0.0.1:6006 root@<server-ip>
```

## 依赖管理

前端统一使用 pnpm：

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm --filter @wm-bench/web build
```

项目根目录的 `package.json` 声明 `pnpm@10.23.0`，锁文件为 `pnpm-lock.yaml`。不要提交 `package-lock.json`。

Python 依赖由启动脚本自动准备。分层入口如下：

- `requirements.txt`: API、Worker 和核心评估依赖入口
- `apps/api/requirements.txt`: FastAPI 服务依赖
- `apps/worker/requirements.txt`: Worker 基础图像处理依赖
- `requirements/evaluator.txt`: 评估算法运行依赖
- `requirements/sharp.txt`: SHARP/3D Viewpoint Re-rendering 可选重型依赖

跳过 SHARP/3D 重型依赖：

```bash
WM_BENCH_INSTALL_SHARP_DEPS=0 bash scripts/start-macos.sh
```

AutoDL 可在 `.env.autodl` 中设置同名变量。

## 资源目录

默认资源位置：

- 数据集: `resources/datasets`
- 模型权重: `resources/weights`
- 资源说明: `resources/README.md`
- 算法说明元数据: `resources/metadata`

AutoDL 默认路径见 `.env.autodl.example`：

- 数据集: `<仓库根目录>/resources/datasets`
- 权重: `<仓库根目录>/resources/weights`
- 运行结果: `/root/autodl-tmp/wm-bench/runs`
- 实验状态索引: `<运行结果目录>/_experiment_state`
- 日志: `/root/autodl-tmp/wm-bench/runs/logs`

攻击算法按 `evaluator/attacks/<folder>` 目录名分类展示。新增水印算法、攻击目录或资源元数据后，重启 API 可刷新资源目录。

## 内置资源清单

以下清单来自当前后端资源注册表和数据集目录配置。`ID` 是实验配置、API 和前端资源页使用的稳定键；真实可运行状态还取决于本机是否已安装对应数据集和权重。

内置数据集 14 项：

| ID | 名称 | 类别 |
| --- | --- | --- |
| `ms-coco` | MS COCO | 基础自然图像基准 |
| `imagenet` | ImageNet | 基础自然图像基准 |
| `diffusiondb` | DiffusionDB | AIGC 图像 |
| `w-bench` | W-Bench | AIGC 图像 |
| `4k-benchmark` | 4K Benchmark Images | 高清版权图 |
| `flickr2k` | Flickr2K | 高清版权图 |
| `openimages-v7` | OpenImages V7 | 真实复杂开放世界图片 |
| `mapillary-vistas` | Mapillary Vistas | 真实复杂开放世界图片 |
| `doclaynet` | DocLayNet | 文档、截图、海报类场景 |
| `publaynet` | PubLayNet | 文档、截图、海报类场景 |
| `shopee-product-matching` | Shopee Product Matching | 电商版权保护 |
| `products-10k` | Products-10K | 电商版权保护 |
| `rico` | RICO | 移动端截图和 UI 内容保护 |
| `mobileviews` | MobileViews | 移动端截图和 UI 内容保护 |

内置水印算法 21 项：

| ID | 名称 | 类型 | 运行设备 |
| --- | --- | --- | --- |
| `chunkyseal` | ChunkySeal | 深度水印 | GPU |
| `cin` | CIN | 深度水印 | GPU |
| `dwsf` | DWSF | 深度水印 | GPU |
| `hidden` | HiDDeN | 深度水印 | GPU |
| `invisible-watermark-dwtdct` | DWT-DCT | 传统/频域水印 | CPU |
| `invisible-watermark-dwtdctsvd` | DWT-DCT-SVD | 传统/频域水印 | CPU |
| `invisible-watermark-rivagan` | RivaGAN | 深度水印 | GPU |
| `invismark` | InvisMark | 深度水印 | GPU |
| `maskwm-d32` | MaskWM-D32 | 深度水印 | GPU |
| `mbrs` | MBRS | 深度水印 | GPU |
| `pimog` | PIMoG | 深度水印 | GPU |
| `pixelseal` | PixelSeal | 深度水印 | GPU |
| `rawatermark` | RAWatermark | 深度水印 | GPU |
| `ssl-watermarking` | SSL Watermarking | 深度水印 | GPU |
| `stegastamp` | StegaStamp | 深度水印 | GPU |
| `traditional-spread-dct` | DCT | 传统/频域水印 | CPU |
| `trustmark-c` | TrustMark-C | 深度水印 | GPU |
| `trustmark-q` | TrustMark-Q | 深度水印 | GPU |
| `videoseal` | VideoSeal | 深度水印 | GPU |
| `vine` | VINE | 深度水印 | GPU |
| `wam` | WAM | 深度水印 | GPU |

内置攻击 43 项：

| 类别 | ID | 名称 | 运行设备 |
| --- | --- | --- | --- |
| Identity | `identity` | Identity | CPU |
| Distortion | `brightness` | Brightness | CPU |
| Distortion | `contrast` | Contrast | CPU |
| Distortion | `erasing` | Erasing | CPU |
| Distortion | `gaussian_blur` | Gaussian Blur | CPU |
| Distortion | `gaussian_noise` | Gaussian Noise | CPU |
| Distortion | `jpeg` | JPEG Compression | CPU |
| Distortion | `resize` | Resize | CPU |
| Distortion | `resized_crop` | Resized Crop | CPU |
| Distortion | `rotation` | Rotation | CPU |
| Consumer Enhancement | `cew_e1` | Auto-Tone | CPU |
| Consumer Enhancement | `cew_e2` | Warm-Vivid | CPU |
| Consumer Enhancement | `cew_e3` | Film-Faded | CPU |
| Consumer Enhancement | `cew_e4` | Local-Clarity HDR | CPU |
| Consumer Enhancement | `cew_d1` | Zero-DCE++ Auto-Light | GPU |
| Consumer Enhancement | `cew_d2` | DeepWB Auto-WhiteBalance | GPU |
| Consumer Enhancement | `cew_d3` | Image-Adaptive 3D LUT | GPU |
| Consumer Enhancement | `cew_d4` | Retinexformer Detail Low-Light Enhance | GPU |
| Consumer Enhancement | `cew_d5` | NAFNet/Restormer AI-Denoise | GPU |
| Consumer Enhancement | `cew_s1` | Real-ESRGAN | GPU |
| Consumer Enhancement | `cew_s2` | SwinIR | GPU |
| Consumer Enhancement | `cew_s3` | BSRGAN | GPU |
| Consumer Enhancement | `cew_c1` | Basic Auto-Fix SR | GPU |
| Consumer Enhancement | `cew_c2` | Color Retouch SR | GPU |
| Consumer Enhancement | `cew_c3` | Detail Enhance SR | GPU |
| Consumer Enhancement | `cew_c4` | Full Enhancement Chain | GPU |
| Physical Channel | `screen_shoot` | PIMoG-style Screen-Camera | CPU |
| Physical Channel | `print_camera` | CamMark-style Print-Camera | CPU |
| Physical Channel | `combined_physical` | Combined Physical Channel | CPU |
| Regeneration | `regen_vae` | CompressAI VAE Reconstruction | GPU |
| Regeneration | `regen_diffusion` | WAVES Diffusion Regeneration | GPU |
| Regeneration | `2x_regen` | 2-pass Diffusion Regeneration | GPU |
| Regeneration | `4x_regen` | 4-pass Diffusion Regeneration | GPU |
| Regeneration | `noise_to_image` | CtrlRegen Noise-to-Image | GPU |
| Regeneration | `image_to_vedio` | NFPA Image-to-Video | GPU |
| 3D Viewpoint Re-rendering | `3d_viewpoint_rerendering_rotate_point` | 3D Viewpoint Rotate (point) | GPU |
| 3D Viewpoint Re-rendering | `3d_viewpoint_rerendering_rotate_ahead` | 3D Viewpoint Rotate (ahead) | GPU |
| 3D Viewpoint Re-rendering | `3d_viewpoint_rerendering_rotate_forward_point` | 3D Viewpoint Rotate Forward (point) | GPU |
| 3D Viewpoint Re-rendering | `3d_viewpoint_rerendering_rotate_forward_ahead` | 3D Viewpoint Rotate Forward (ahead) | GPU |
| 3D Viewpoint Re-rendering | `3d_viewpoint_rerendering_shake_point` | 3D Viewpoint Shake (point) | GPU |
| 3D Viewpoint Re-rendering | `3d_viewpoint_rerendering_shake_ahead` | 3D Viewpoint Shake (ahead) | GPU |
| 3D Viewpoint Re-rendering | `3d_viewpoint_rerendering_swipe_point` | 3D Viewpoint Swipe (point) | GPU |
| 3D Viewpoint Re-rendering | `3d_viewpoint_rerendering_swipe_ahead` | 3D Viewpoint Swipe (ahead) | GPU |

## 环境配置

本地开发可复制 `.env.example` 为 `.env`：

```bash
cp .env.example .env
```

AutoDL 首次部署会在缺失时从 `.env.autodl.example` 生成 `.env.autodl`。

常用环境变量：

```bash
APP_ENV=autodl
API_HOST=0.0.0.0
API_PORT=6006
WM_BENCH_RESOURCES_ROOT=./resources
WM_BENCH_RUNS_ROOT=/root/autodl-tmp/wm-bench/runs
WM_BENCH_DEVICE=cuda:0
NEXT_PUBLIC_API_BASE_URL=
```

`NEXT_PUBLIC_API_BASE_URL` 留空时，前端按同源 API 访问，适合 AutoDL 或反向代理部署。如果前端和 API 分开部署，需要设置为 API 公网地址，并在 `WM_BENCH_CORS_ORIGINS` 中加入前端 origin。

## 验证

部署前检查：

```bash
python3 scripts/check-deploy-readiness.py
```

该脚本会检查项目根目录、资源目录、数据集目录、权重目录、运行目录、实验状态目录、算法/攻击目录扫描和 Worker 心跳。服务运行后也可以访问 `/system/readiness` 查看同一份检查结果。

前端生产构建：

```bash
pnpm --filter @wm-bench/web build
```

资源目录单元测试：

```bash
python3 -m unittest apps.api.tests.test_resource_catalog
```

AutoDL 状态检查：

```bash
bash scripts/deploy-autodl-linux.sh status
```