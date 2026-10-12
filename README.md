# GeosunPre

原生 C++17/CUDA 双鱼眼与 LAS 重建程序。单进程完成语义掩码、LoMa 特征与匹配、自有 SfM、共享鱼眼内参优化、LAS 配准和 COLMAP/PLY 导出。运行不需要 Python、LibTorch、COLMAP、WSL 或联网。

## 构建

Windows x64：Visual Studio 2022 C++ 桌面工具链、Windows SDK、CUDA Toolkit 12.8、CMake 3.24 以上。

```powershell
.\build.ps1
```

首次配置下载固定版本依赖到 `build/_deps/`。Ceres 使用内置轻量日志与 Eigen 稀疏求解，不引入 SuiteSparse、METIS 或 BLAS。OpenCV 只构建 core、imgproc、imgcodecs、calib3d 及后者必要的 features2d/flann，不构建 GUI、Python 绑定或测试框架。

## 模型

将符合[模型契约](docs/models.md)的四个 TensorRT 引擎、清单与推理 DLL 放入 `models/`。资源不进入 Git。引擎为实际 GPU 构建，TensorRT 固定为 10.13.3.9；导出是独立资源准备，工程不包含 Python 转换环境。

```text
models/
  manifest.json
  engines/{detector,descriptor,matcher,segmentation}.engine
  runtime/nvinfer_10.dll
```

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

`.\package.ps1` 生成 `dist/` 独立运行包：EXE、CUDA/模型推理运行库、引擎和硬文档。已存在时先移走旧包。

约定：[输入与坐标](docs/input.md)、[输出](docs/output.md)、[模型](docs/models.md)、[缓存](docs/cache.md)。
