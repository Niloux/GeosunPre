# GeosunPre

原生 C++17/CUDA 双鱼眼与 LAS 重建程序。单进程完成语义掩码、LoMa 特征与匹配、自有 SfM、共享鱼眼内参优化、LAS 配准和 COLMAP/PLY 导出。运行不需要 Python、LibTorch、COLMAP、WSL 或联网。

## 开始使用

在 Windows x64 上安装 Visual Studio 2022 C++ 桌面工具链、Windows SDK、CUDA Toolkit 12.8、CMake 3.24 以上，然后在项目根目录执行：

```powershell
.\build.ps1
.\prepare-models.ps1
.\run.ps1 run --input "D:/数据/场景" --work "work/场景"
```

这就是完整的默认流程。构建始终使用当前源码；GPU 预处理、特征颜色缓存、鱼眼 PnP、解析 BA 导数和迭代 Schur 优化均默认启用。

| 步骤 | 默认行为 |
| --- | --- |
| `build.ps1` | 编译 Release 程序到 `build/bin/Release/geosunpre.exe`，自动应用 Ceres Schur 回代修复 |
| `prepare-models.ps1` | 为 GPU 0 准备 `models/`，匹配器默认 FP16，原图尺寸默认宽 3000、高 4000 |
| `run.ps1 run` | 使用上述程序和 `models/`，完成重建、LAS 配准及 COLMAP/PLY 导出 |

首次构建下载固定依赖到 `build/_deps/`；首次准备模型下载转换工具和固定权重到 `.cache/model-prepare/`，需要联网及数 GB 下载空间。此后复用缓存。模型准备使用独立 Python 环境，运行程序与独立运行包使用 C++/CUDA/TensorRT。

## 日常运行

```powershell
.\run.ps1 run --work "work/场景"
.\run.ps1 prepare --input "D:/数据/场景" --work "work/场景"
.\run.ps1 export --work "work/场景" --output "D:/成果/场景"
.\run.ps1 check --output "D:/成果/场景"
.\run.ps1 info
```

默认交付 `WORK/colmap/`，包含原图、掩码、鱼眼 COLMAP 模型和 5 cm LAS 初始化 PLY。`--no-ply` 省略 PLY，`--ply-voxel` 调整体素边长。完整结构见[输出约定](docs/output.md)。

阶段为 `prepare → masks → features → matches → sfm → align → export`。`run --work WORK` 复用已完成阶段并恢复检查点；`run --work WORK --stage 阶段` 单独执行指定阶段，须先完成前置阶段。`prepare` 检查真实输入、相机和导航转换，不加载模型。

默认配置见 [configs/default.json](configs/default.json)，可通过 `--config` 指定。内参标定默认使用全部有效轨迹（`calibration_tracks=all`），随后锁定内参补全轨迹并精调。标定选项及坐标约定见[输入文档](docs/input.md)。

`run.ps1` 固定启动 `build/bin/Release/geosunpre.exe`，启动日志和 `info` 显示程序路径、构建版本及时间。重新编译只需执行 `build.ps1`。

## 模型与工作目录

默认模型目录为项目根目录下的 `models/`，TensorRT 固定为 10.13.3.9：

```text
models/
  manifest.json
  engines/{detector,descriptor,matcher,segmentation}.engine
  runtime/nvinfer_10.dll
```

匹配器默认 FP16，Softmax、Reduce 和 Normalization 层保留 FP32；所有浮点输入输出保持 float32。模型准备后自动用真实照片验证四模型输出、动态描述子数量和匹配补零行为。图像尺寸、设备和精度定制参数见[模型文档](docs/models.md)。

模型目录中完整且兼容的包会校验后复用。更换 GPU、图像尺寸或匹配精度时使用新模型目录，并用 `--models` 指定；新生成的目录同时保留 ONNX 和准备元数据。模型资源不进入 Git。

工作目录保存冻结的输入、配置和模型位置。输入、配置或模型改变时使用新工作目录；续跑使用工作目录中记录的模型。不同请求的交付包拒绝覆盖。缓存格式与实现标记须匹配，详见[缓存约定](docs/cache.md)。

进度在 `state.json`，最终结果在 `result.json`。`cache/*_profile.json` 和 `cache/sfm.json` 保存前端耗时、BA 收敛、轨迹和几何质量统计，`result.json` 同时保存联合配准各项验收结果。

## 打包部署

完成模型准备后，编译并打包：

```powershell
.\build.ps1
.\package.ps1
```

`package.ps1` 将本次编译的程序、`models/` 中的引擎、运行库和文档复制到 `dist/`，并用真实照片执行四模型检查，通过后发布。支持 `-Models`、`-Output`、`-Image`；输出目录须不存在。打包只携带运行资源。

将整个 `dist/` 目录复制到模型所对应 GPU 的 Windows 机器，按包内 README 运行。独立 EXE 默认读取同级 `models/`，详细说明见[部署文档](docs/deployment.md)。

构建依赖使用固定版本；Ceres 使用轻量日志和 Eigen 稀疏求解，OpenCV 只构建图像处理所需模块。Ceres 回代补丁由 CMake 自动应用，清空 `build/` 后重新构建同样生效。
