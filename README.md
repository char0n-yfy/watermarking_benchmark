# WaterPrism

**WaterPrism: 面向图像鲁棒水印的综合智能测评系统**

WaterPrism 是一个面向图像鲁棒水印算法的综合测评平台。系统以 Web 界面组织数据集、水印算法、攻击方法、实验配置、运行队列和结果展示，由 FastAPI 提供后端服务，本地 Worker 执行水印嵌入、攻击、检测和质量评估任务。

正式实验推荐运行在 Linux / AutoDL。macOS 和 Windows 可用于资源查看、配置编辑、界面验收和 CPU 子集实验；部分 CUDA、3D/SHARP 或大模型依赖在本地桌面系统上可能不可用。

## 系统能力

- 数据集、水印算法、攻击方法的统一资源目录。
- Web 化实验配置、运行队列、Worker 心跳和运行状态监控。
- 覆盖传统失真、消费级增强、物理信道、再生成、3D 视角重渲染等攻击类型。
- 支持本地开发运行，也支持 AutoDL 单机生产式部署。
- AutoDL 部署时前端静态页面由 FastAPI 同端口托管，只需要暴露一个服务端口。

## 目录结构

```text
apps/
  api/        FastAPI 后端服务
  web/        Next.js 前端，生产构建输出到 apps/web/out
  worker/     本地实验执行 Worker
evaluator/    水印算法、攻击算法和评估逻辑
scripts/      唯一服务入口、依赖初始化和维护工具
docs/         评分协议和开发规范
resources/    数据集、权重和资源元数据入口
requirements/ Python 分层依赖
runs/         运行结果、日志和实验状态
```

## 统一服务入口

三种 OS 共用同一个主入口 `scripts/wmbench.py`，不再由平台脚本各自实现服务生命周期：

```bash
python3 scripts/wmbench.py --profile local up
python3 scripts/wmbench.py --profile local status
python3 scripts/wmbench.py --profile local logs
python3 scripts/wmbench.py --profile local restart
python3 scripts/wmbench.py --profile local down
```

Windows 将 `python3` 换成 `py -3`。所有平台直接调用该入口，不再提供独立的 Shell 或 PowerShell 启停脚本。主入口支持三种运行模式：

| Profile | 用途 | Web 运行方式 | 默认端口 |
| --- | --- | --- | --- |
| `local` | macOS、Windows、Linux 本地开发 | Next.js dev server | Web 3000 + API 8000 |
| `production` | 三种 OS 的生产式单机运行 | 静态构建由 FastAPI 托管 | 6006 |
| `autodl` | AutoDL GPU 部署 | 静态构建由 FastAPI 托管 | 6006 |

`--profile auto` 是默认值：AutoDL 主机选择 `autodl`，其他机器选择 `local`。`up` 会按指纹增量准备依赖和生产构建；已经完成 bootstrap 时可用 `up --no-bootstrap` 快速启动。

配置优先级固定为：命令行参数 > 当前环境变量 > profile dotenv > 内置默认值。`local`、`production`、`autodl` 分别使用 `.env`、`.env.production`、`.env.autodl`；后两者缺失时从对应 example 自动创建。

常用参数示例：

```bash
python3 scripts/wmbench.py --profile local --api-port 8001 --web-port 3001 --device mps up
python3 scripts/wmbench.py --profile production --api-port 6006 up
python3 scripts/wmbench.py --profile autodl --skip-sharp up
```

生产式部署启动后，Web UI 和 API 同源，只需开放 API 端口。远程服务器推荐 SSH 隧道：

```bash
ssh -L 6006:127.0.0.1:6006 root@<server-ip>
```

各环境的启动命令为：

| 环境 | 启动命令 |
| --- | --- |
| macOS / Linux 本地开发 | `python3 scripts/wmbench.py --profile local up` |
| Windows 本地开发 | `py -3 scripts/wmbench.py --profile local up` |
| 通用生产式单机部署 | `python3 scripts/wmbench.py --profile production up` |
| AutoDL | `python3 scripts/wmbench.py --profile autodl up` |

停止时将 `up` 换成 `down`。Windows 如需准确的 CPU 功耗读数，可在 `.env` 中设置 `WM_BENCH_LHM_PATH`；统一入口会自动启动 LibreHardwareMonitor，设置 `WM_BENCH_SKIP_LHM=1` 可关闭该行为。

## 依赖管理

前端统一使用 pnpm：

```bash
corepack enable
pnpm install --frozen-lockfile
pnpm --filter @wm-bench/web build
```

项目根目录的 `package.json` 声明 `pnpm@10.23.0`，锁文件为 `pnpm-lock.yaml`。项目不使用 `package-lock.json`。

Python 依赖由统一入口的 `bootstrap`/`up` 增量准备。分层入口如下：

- `requirements.txt`: API、Worker 和核心评估依赖入口
- `apps/api/requirements.txt`: FastAPI 服务依赖
- `apps/worker/requirements.txt`: Worker 基础图像处理依赖
- `requirements/evaluator.txt`: 评估算法运行依赖
- `requirements/sharp.txt`: SHARP/3D Viewpoint Re-rendering 可选重型依赖

跳过 SHARP/3D 重型依赖：

```bash
python3 scripts/wmbench.py --profile local --skip-sharp up
```

AutoDL 可在 `.env.autodl` 中设置同名变量。

## 资源准备

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

新增数据集、权重、水印算法、攻击目录或资源元数据后，重启 API 可刷新资源目录。

## 内置资源清单

以下清单来自当前后端资源注册表和数据集目录配置。`ID` 是实验配置、API 和前端资源页使用的稳定键；真实可运行状态取决于本机是否已安装对应数据集和权重。

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

生产式部署和 AutoDL 首次运行会分别从 `.env.production.example`、`.env.autodl.example` 生成 `.env.production`、`.env.autodl`。

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

## 验收检查

建议按以下顺序验收：

1. 服务只由 `scripts/wmbench.py` 启动和关闭，不存在另一套 OS 专用生命周期实现。
2. Web UI 可以打开，API 健康检查返回正常：本地开发默认 `http://127.0.0.1:8000/health`，AutoDL 默认 `http://127.0.0.1:6006/health`。
3. 资源页可以展示内置数据集、水印算法和攻击列表，并能识别本机已安装的数据集和权重。
4. 部署前检查返回 ready，或仅在缺少可选资源时给出可解释的 WARN。
5. 前端生产构建通过。
6. 资源目录单元测试通过。
7. AutoDL 生产部署只暴露 `API_PORT` 一个端口，默认 `6006`，前端和 API 同源访问。

统一部署前检查：

```bash
python3 scripts/wmbench.py --profile autodl check
```

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
python3 scripts/wmbench.py --profile autodl status
```

## 安全说明

当前系统没有登录鉴权。不要把服务直接裸露给不可信公网用户。正式使用时优先选择 AutoDL 隧道、SSH 隧道、VPN，或带访问控制的反向代理。
