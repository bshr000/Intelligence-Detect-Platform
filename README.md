# YOLO-CMFM Platform

基于 FastAPI 与 Next.js 的可见光–SAR 多模态遥感目标检测推理平台。仓库内置整理后的
YOLOv11-CMFM 源码，后端通过外层适配器复用原四通道推理流程，不修改模型核心代码。

## 目录

```text
backend/                    FastAPI 推理 API
frontend/                   Next.js 可视化平台
third_party/YOLOv11-CMFM/   原始模型、CMFM 模块与配置
weights/                    本地模型权重（不进入 Git）
docker-compose.yml          前后端容器编排
```

模型适配方式如下：

1. API 接收 `rgb_image` 与 `sar_image`。
2. 校验两幅图像可解码且尺寸一致。
3. 将图像暂存为同名的 `visible/input.png` 与 `infrared/input.png`。
4. 调用原流程 `YOLO(...).predict(use_simotm="RGBT", channels=4)`。
5. 返回检测 JSON 和可视化结果 PNG，随后删除请求临时文件。

## 准备权重

将真实权重放到：

```text
weights/best.pt
```

也可通过环境变量传入其他路径：

```powershell
$env:MODEL_WEIGHTS = 'D:\path\to\best.pt'
```

权重、上传文件、推理结果、`node_modules` 和 `.next` 均已在 `.gitignore` 中排除。

## 本地启动

建议使用与训练一致的 CUDA、PyTorch 环境。以下命令在仓库根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .\third_party\YOLOv11-CMFM
python -m pip install -r .\backend\requirements-dev.txt
python -m uvicorn app.main:app --app-dir backend --reload --port 8000
```

另开终端启动前端：

```powershell
cd frontend
pnpm install --frozen-lockfile
Copy-Item .env.example .env.local
pnpm dev
```

访问：

- Web：<http://localhost:3000>
- API 文档：<http://localhost:8000/docs>
- 模型就绪检查：<http://localhost:8000/api/v1/health/ready>

未配置权重时 API 仍能启动，但就绪检查返回 HTTP 503，真实推理不可用。前端开发如需 Mock，
将 `NEXT_PUBLIC_USE_MOCK_API` 设为 `true`。

## Docker

默认后端镜像使用 PyTorch 2.4.1、CUDA 11.8，并需要主机已配置 NVIDIA Container Toolkit：

```powershell
docker compose up --build
```

## 测试

```powershell
python -m pytest backend\tests
cd frontend
pnpm lint
pnpm build
```

无权重 CI 只验证 API 生命周期、接口契约和前端构建。真实推理应在有 GPU 与权重的环境中进行
端到端验证。

## API

- `GET /api/v1/health/live`
- `GET /api/v1/health/ready`
- `GET /api/v1/models/current`
- `POST /api/v1/detections`
- `GET /api/v1/results/{request_id}.png`

推理接口使用 `multipart/form-data`，主要字段为：

- `rgb_image`
- `sar_image`
- `confidence`，默认 `0.25`
- `iou`，默认 `0.70`
- `imgsz`，默认 `640`

## 许可证

仓库包含 AGPL-3.0 许可的 Ultralytics/YOLOv11-CMFM 衍生代码，因此本项目使用 AGPL-3.0。
第三方来源说明见 `THIRD_PARTY_NOTICES.md`。

