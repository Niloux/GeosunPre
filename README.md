# GeosunPre

将同一套 Geosun 异步 L/R 双鱼眼图像、导航位置和 LAS 转为 3DGS 训练数据。每次曝光独立求位姿，L/R 各共享一组原生 `OPENCV_FISHEYE` 内参；自动选择标定支撑，无需填写场景帧号。

## 启动

Windows 默认使用独立 C++/CUDA 流程，生产运行不启动 Python 或 WSL。Linux 和显式参考流程保留 Python 3.13。需要 NVIDIA CUDA GPU 和本地模型权重，准备方式见[资源说明](docs/resources.md)。

Windows 开发构建使用 Visual Studio C++ 工具链、CUDA Toolkit 12.8 和 vcpkg。`setup-windows.ps1` 准备开发/参考环境，`setup-native.ps1` 构建原生程序；固定权重只需转换一次，生产运行加载转换后的 `.pt`。

通过 `VCPKG_ROOT` 指定 vcpkg，或在项目同级放置 vcpkg。程序和依赖位于 `.cache/native/`，`package-native.ps1` 整理可独立使用的 EXE、DLL 和权重目录。实现和验收进度见 [native](native/README.md)与[迁移记录](native/MIGRATION.md)。旧工作目录和结果保留，原生任务使用新的工作目录。

```powershell
.\setup-windows.ps1
.\run.ps1 --input "input/场景" --work "work/场景-native"
.\run.ps1 --work "work/场景-native"

# 仅检查依赖导入、CUDA、原生扩展和 C++ 求解程序
.\setup-windows.ps1 -CheckOnly

# 导出与验包
.\.cache\native\build\bin\geosunpre-native.exe export --work "work/场景-native" --output "D:/output/colmap"
.\.cache\native\build\bin\geosunpre-native.exe check --output "D:/output/colmap"

# 继续原来的 Python 任务
.\run-reference.ps1 --work "work/场景"
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

`--image-mode link` 适合本机使用；迁移数据使用默认 `copy`。原生最终选择见 `WORK/registration/native/report.json`，质量见 `WORK/colmap/quality.json`，运行状态见 `WORK/pipeline_status.json`。参考流程保留原来的 HTML 检查页。

## 文档

| 文档 | 内容 |
| --- | --- |
| [生产运行](docs/production_pipeline.md) | 配置、阶段、续跑与导出 |
| [算法与架构](docs/architecture.md) | 自动策略、模块职责与尚待验证的问题 |
| [COLMAP 契约](docs/colmap_contract.md) | 模型、坐标、掩码、初始化 PLY 与验包 |
| [资源与离线运行](docs/resources.md) | 依赖、权重缓存与资源读数 |
| [2026-10-09 验证](docs/benchmarks/generalization_20261009.md) | 本轮实际运行、证据范围与历史记录 |
| [Windows 原生数值验证](docs/benchmarks/windows_native_20261012.md) | 性能归因、同模型 BA 对照、实际交付与迁移进度 |
| [Windows 完整 C++/CUDA 验证](docs/benchmarks/windows_cpp_pipeline_20261012.md) | 从原始输入执行的原生整场任务、部署和交付验收 |

原始数据、权重和运行产物位于忽略跟踪的 `input/`、`.cache/`、`work/`。

开发协作约定见 [AGENTS.md](AGENTS.md)。
