# GeosunPre

将同一套 Geosun 异步 L/R 双鱼眼图像、导航位置和 LAS 转为 3DGS 训练数据。每次曝光独立求位姿，L/R 各共享一组原生 `OPENCV_FISHEYE` 内参；自动选择标定支撑，无需填写场景帧号。

## 启动

支持 Linux 和 Windows 本机，需要 Python 3.13、NVIDIA CUDA GPU、C++17 编译器和本地模型权重。依赖版本由 `uv.lock` 固定，权重准备见[资源说明](docs/resources.md)。

Windows 在 PowerShell 中运行；首次配置会复用已安装的 `uv`、自动找到 Visual Studio C++ 工具链，并在项目内创建独立环境：

```powershell
.\setup-windows.ps1
.\run.ps1 --input "input/场景"
.\run.ps1 --work "work/场景"

# 仅检查依赖导入、CUDA 和原生扩展
.\setup-windows.ps1 -CheckOnly

# 导出与验包
.\.venv\Scripts\python.exe -X utf8 -m geosunpre export --work "work/场景" --output "D:/output/colmap"
.\.venv\Scripts\python.exe -X utf8 -m geosunpre check --output "D:/output/colmap"
```

Linux：

```bash
uv sync --locked

# 新任务；默认工作目录 work/场景
./run.sh --input input/场景

# 继续已有任务
./run.sh --work work/场景

# 导出到其他目录并独立验包
.venv/bin/geosunpre export --work work/场景 --output /path/to/colmap
.venv/bin/geosunpre check --output /path/to/colmap
```

可用 `--work` 指定工作目录、`--config` 覆盖资源和验收限值，示例见 [configs/default.toml](configs/default.toml)。运行离线加载权重。

## 输入

```text
input/场景/
  场景.pro
  场景_CameraPos_C2E.txt
  场景_pos.txt
  场景_pos_ENH.txt
  ResultOut/场景.las
  Image/L/0000/*.JPG
  Image/R/0000/*.JPG
```

只使用 CameraPos 有精确同名记录的 L/R 图像，M 不参与。文件名为 `GPS周-秒-毫秒_帧号-L或R.JPG`；同帧号不表示同步曝光。尺寸必须符合 `.pro`。LAS 支持未压缩 1.2、点格式 0–3，多份 LAS 用 `registration_las` 指定。

## 成果

最终训练数据位于 **`WORK/colmap/`**，默认复制原图，附带有效域掩码和 5 cm LAS 初始化点云 `sparse/0/points3D.ply`。`points3D.bin` 保存 SfM 稀疏点及轨迹。图像不缩放、不去畸变，下游须支持原生鱼眼模型。

`--image-mode link` 适合本机使用；迁移数据使用默认 `copy`。最终选择与质量见 `WORK/registration/final/inspection.html`，运行状态见 `WORK/pipeline_status.json`。

## 文档

| 文档 | 内容 |
| --- | --- |
| [生产运行](docs/production_pipeline.md) | 配置、阶段、续跑与导出 |
| [算法与架构](docs/architecture.md) | 自动策略、模块职责与尚待验证的问题 |
| [COLMAP 契约](docs/colmap_contract.md) | 模型、坐标、掩码、初始化 PLY 与验包 |
| [资源与离线运行](docs/resources.md) | 依赖、权重缓存与资源读数 |
| [2026-10-09 验证](docs/benchmarks/generalization_20261009.md) | 本轮实际运行、证据范围与历史记录 |

原始数据、权重和运行产物位于忽略跟踪的 `input/`、`.cache/`、`work/`。

开发协作约定见 [AGENTS.md](AGENTS.md)。
