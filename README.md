# Watermarking Benchmark

图像水印鲁棒性 Benchmark Web MVP。当前主部署路径是本地/AutoDL 单机运行：FastAPI 负责资源目录、实验配置、运行队列和静态前端；本地 worker 轮询队列并执行实验。

## 快速启动

macOS / 本地开发：

```bash
bash scripts/start-macos.sh
```

Linux / AutoDL：

```bash
bash scripts/start-autodl-linux.sh
```

启动后打开脚本输出中的 Web URL。AutoDL 默认使用 `6006`，FastAPI 会在同一端口服务 `apps/web/out` 静态前端，浏览器和 API 保持同源访问。

## 资源目录

大文件不进入 Git。下载或解压后放到这些目录：

- 数据集：`resources/datasets`
- 模型权重：`resources/weights`
- 运行结果：`runs/local` 或 `WM_BENCH_RUNS_ROOT`

服务器/AutoDL 默认路径见 `.env.autodl.example`：

- 数据集：`/root/autodl-fs/wm-bench/resources/datasets`
- 权重：`/root/autodl-fs/wm-bench/resources/weights`
- SQLite：`/root/autodl-fs/wm-bench/state/wmbench.sqlite`
- 运行结果：`/root/autodl-tmp/wm-bench/runs`

攻击算法在前端按 `evaluator/attacks/<folder>` 目录名分类展示；新增攻击目录后，重启 API 即可刷新资源目录。

## 部署注意

常用环境变量：

```bash
APP_ENV=autodl
API_HOST=0.0.0.0
API_PORT=6006
WM_BENCH_RESOURCES_ROOT=/root/autodl-fs/wm-bench/resources
WM_BENCH_RUNS_ROOT=/root/autodl-tmp/wm-bench/runs
WM_BENCH_DB_PATH=/root/autodl-fs/wm-bench/state/wmbench.sqlite
NEXT_PUBLIC_API_BASE_URL=
```

`NEXT_PUBLIC_API_BASE_URL` 留空时，前端按同源 API 访问，适合 AutoDL/反向代理部署。如果前端和 API 分开部署，需要把它设置为 API 公网地址，并在 `WM_BENCH_CORS_ORIGINS` 中加入前端 origin。

当前没有登录鉴权。不要把服务直接裸露给不可信公网用户；优先使用 AutoDL 隧道、SSH 隧道、VPN，或放在带访问控制的反向代理之后。

部署前检查：

```bash
python3 scripts/check-deploy-readiness.py
```

该脚本会检查资源目录、数据集目录、权重目录、运行目录可写性、SQLite、算法/攻击目录扫描和 worker 心跳。缺数据集或 worker 未启动会显示 WARN；SQLite、运行目录、资源目录扫描失败会显示 FAIL。服务运行后也可以访问 `/system/readiness` 查看同一份检查结果。

## 验证

前端构建：

```bash
pnpm --filter @wm-bench/web build
```

资源目录测试：

```bash
python3 -m unittest apps.api.tests.test_resource_catalog
```
