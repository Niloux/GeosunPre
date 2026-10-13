# GeosunPre

原生 C++17/CUDA 双鱼眼与 LAS 重建程序。单进程完成语义掩码、LoMa 特征与匹配、自有 SfM、共享鱼眼内参优化、LAS 配准和 COLMAP/PLY 导出。运行不需要 Python、LibTorch、COLMAP、WSL 或联网。

## 构建

Windows x64：Visual Studio 2022 C++ 桌面工具链、Windows SDK、CUDA Toolkit 12.8、CMake 3.24 以上。

```powershell
.\build.ps1
```

首次配置下载固定版本依赖到 `build/_deps/`。Ceres 使用内置轻量日志与 Eigen 稀疏求解，不引入 SuiteSparse、METIS 或 BLAS。OpenCV 只构建 core、imgproc、imgcodecs、calib3d 及后者必要的 features2d/flann，不构建 GUI、Python 绑定或测试框架。

## 模型

运行一次模型准备，再编译和运行：

```powershell
.\prepare-models.ps1
.\build.ps1
.\run.ps1 run --input "D:/数据/场景" --work "work/场景" --models models
```

`prepare-models.ps1` 自动在 `.cache/model-prepare/` 安装独立转换环境，下载固定权重、导出 ONNX、为 GPU 0 构建四个 TensorRT 引擎，并用真实照片检查输出、动态描述子数量和匹配补零行为。首次需要联网，工具和权重下载为数 GB；之后复用缓存。已有匹配当前 GPU、版本和图像尺寸的完整模型包只校验并复用。

默认原图尺寸为宽 3000、高 4000，可用 `-ImageWidth`、`-ImageHeight` 指定实际尺寸。更换 GPU 或尺寸时使用新模型目录；相同尺寸可复用已有 ONNX，省去权重导出：

```powershell
.\prepare-models.ps1 -Models models/new-gpu -OnnxFrom models
.\prepare-models.ps1 -Models models/new-size -ImageWidth 4000 -ImageHeight 3000
.\prepare-models.ps1 -Models models/fast -OnnxFrom models -MatcherPrecision fp16
```

资源不进入 Git。TensorRT 固定为 10.13.3.9，完整参数与固定来源见[模型契约](docs/models.md)。Python 仅用于独立资源准备，原生程序和 `dist/` 运行包不使用它。

```text
models/
  manifest.json
  engines/{detector,descriptor,matcher,segmentation}.engine
  runtime/nvinfer_10.dll
```

新生成的目录还保留 `onnx/` 和准备元数据；`package.ps1` 只复制运行所需的清单、引擎及推理 DLL。

## 使用

```powershell
.\run.ps1 run --input "D:/数据/场景" --work "work/场景" --models models
.\run.ps1 run --work "work/场景"
.\run.ps1 prepare --input "D:/数据/场景" --work "work/场景"
.\run.ps1 export --work "work/场景" --output "D:/成果/场景"
.\run.ps1 check --output "D:/成果/场景"
```

默认交付 `WORK/colmap/`，复制原图并附带掩码和 5 cm LAS 初始化 PLY。`--config configs/default.json` 设置资源与验收限值，`--no-ply` 省略 PLY，`--ply-voxel` 调整体素边长。

阶段为 `prepare → masks → features → matches → sfm → align → export`。`run --work WORK --stage 阶段` 执行指定阶段，前置阶段须已完成。输入、配置或模型变化时使用新工作目录，不兼容旧 Python/LibTorch work。

`prepare` 不加载模型，检查真实输入、相机和导航转换。进度在 `state.json`，最终结果汇总为 `result.json`。不同请求的现有交付包拒绝覆盖。

根目录 `run.ps1` 固定运行 `build/bin/Release/geosunpre.exe`，不会被旧 `dist/` 覆盖；启动时显示程序路径、构建版本和时间。独立 EXE 默认使用同级 `models/`，开发脚本为本进程设置项目模型目录，续跑使用工作目录记录的模型位置。

`.\package.ps1` 生成 `dist/` 独立运行包：EXE、CUDA/模型推理运行库、引擎和硬文档。支持 `-Models` 和 `-Output` 指定新包，已有目录拒绝覆盖。发布前用真实照片执行四个模型；默认使用准备模型时下载的公开 LoMa 样例，也可传 `-Image`。照片不进入发布包。包内 README 使用独立 EXE 命令，`check-models --image FILE` 可在目标机器检查真实推理。

当前实现标记为 `pipeline_version=6`，旧版 work 需换用新目录，原成果仍可独立回读检查。前端使用整数双三次 GPU 缩放、GPU 掩码融合和有界读写队列。配对保留密集近邻和间隔连接，并按曝光序列分段检索、抑制同镜头的相近曝光；仅针对未注册或观测覆盖不足的图像加密补边。特征配额和几何验收保持不变。SfM 的两视图和 PnP 均按鱼眼像素误差判断，使用解析 BA 导数和迭代 Schur，颜色随特征缓存。末尾先标定、锁定内参补全与合并轨迹，再精调。配置 `calibration_tracks` 可选 `all`（默认）或 `stable`；后者以至少三视图、足够视差且覆盖完整的轨迹标定，再锁定内参优化全部点。`cache/*_profile.json`、`cache/sfm.json` 和 `result.json` 保存细项耗时、优化收敛与联合配准各项验收结果；质量报告附带轨迹长度、主点变化与分角度误差。

约定：[输入与坐标](docs/input.md)、[输出](docs/output.md)、[模型](docs/models.md)、[缓存](docs/cache.md)。
