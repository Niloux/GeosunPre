# GeosunPre

原生 C++17/CUDA 双鱼眼与 LAS 输入预处理程序，交付包是定制 LichtFeld Studio（LFS）3DGS 训练的数据目录。单进程完成语义掩码、LoMa 特征与匹配、自有 SfM、共享鱼眼内参优化、LAS 配准、卷帘快门校正、稀疏点清理、COLMAP/PLY 及深度和法向监督导出（默认由 MoGe-2 生成、LAS 定尺度）。运行不需要 Python、LibTorch、COLMAP、WSL 或联网。

## 开始使用

项目仅支持 Windows x64。安装 Visual Studio 2022 C++ 桌面工具链及 CMake 工具、Windows SDK、CUDA Toolkit 12.8、CMake 3.24 以上。首次使用时在项目根目录准备本机模型（已有 `models/` 可跳过）：

```powershell
.\run.ps1 prepare-models sam3.pt
```

| 用途 | 命令 |
| --- | --- |
| 本地调试 | `.\run.ps1 run --input "D:/数据/场景" --output "D:/成果/场景"`，自动增量编译并使用本机模型 |
| 发给别人 | `.\package.ps1`，提交源码后执行，自动编译并打包跨显卡模型 |

构建使用 Ninja Release，首次下载固定依赖到 `build/ninja/_deps/`，之后复用缓存。需要单独调整并行数或强制重新配置时，可调用 `tools/windows/build.ps1` 的 `-Jobs` 或 `-Configure`。

首次准备模型下载转换工具和固定权重到 `.cache/model-prepare/`，需要联网及数 GB 下载空间。此后复用缓存。模型准备使用独立 Python 环境，运行程序与独立运行包使用 C++/CUDA/TensorRT。

## 日常运行

```powershell
.\run.ps1 run --input "D:/数据/场景" --output "D:/成果/场景"
.\run.ps1 run --input "D:/数据/场景" --output "D:/成果/场景" --work "work/场景" --debug
.\run.ps1 run --work "work/场景"
.\run.ps1 export --work "work/场景" --output "D:/成果/场景-重新导出"
.\run.ps1 check --output "D:/成果/场景"
.\run.ps1 info
```

交付直接写入 `--output` 根目录（默认 `output/<输入目录名>`）。

**生产模式**（默认）：工作目录及缓存放在 `OUT.work`（或 `--work` 指定的目录），交付发布并回读校验后整个删除，最后只剩交付目录。中途失败时保留工作目录，以相同命令重跑即可续跑，成功后再删除。交付目录已存在而没有对应工作目录时拒绝重跑。

**调试模式**（`--debug`）：工作目录（默认 `work/<输入目录名>`）及 `cache/` 全部保留，可 `run --work WORK` 续跑或 `--stage` 单独执行阶段，也可用于独立 `export`。模式在首次运行时记入 `state.json`；生产工作目录续跑时加 `--debug` 可改为保留，调试工作目录不会被删除。

交付包含照片（采用卷帘校正时另附读出运动参数）、掩码、鱼眼 COLMAP 模型、50 m 视锥内分档体素 LAS 初始化 PLY（基准体素默认 5 cm，上限 250 万点），以及深度和法向监督图：默认最长边 1024 px，由 MoGe-2 单路 90° 子图同时估计深度和法向，按法向与之一致的可信 LAS 定尺度；`--supervision-source las` 改为 LAS 投影监督。`--no-ply` 省略 PLY，`--ply-voxel` 调整基准体素边长；`--no-las-supervision` 省略深度、法向图和 `supervision.json`。开着 LAS 监督时，持续出现的人物（画框、海报里的人，一直坐着不动的人），以及有 LAS 表面支撑的天空误检块和封闭室内场景残余天空误检均在掩码里改回静态 `255`（配合 `configs/trainer.json` 统一的 `segment_and_ignore`），`--no-static-masks` 保留 SAM3 掩码。完整结构见[输出约定](docs/output.md)。

监督尺寸、额外几何掩码、显存预算及 LFS 训练示例见[深度与法向监督](docs/supervision.md)。已有调试工作目录可直接导出到新目录，无需重跑重建。

默认阶段为 `prepare → masks → features → matches → sfm → align → shutter → cleanup → export`。配置 `"rolling_shutter": false` 去掉卷帘校正阶段，默认 `true`。配置 `"enhanced_photos": true` 改用 `Image/enhanced` 照片，默认 `false` 只用原图。这些选择随配置冻结，续跑自动沿用，改变选择须使用新工作目录。

`run --work WORK` 复用已完成阶段并恢复检查点；`run --work WORK --stage 阶段` 单独执行指定阶段，须先完成前置阶段。`prepare` 检查真实输入、相机和导航转换，不加载模型。

阶段产物、配置范围、导航匹配、卷帘快门和验收规则见[重建流程与算法](docs/reconstruction.md)。

默认流程的最终所选模型在 `cache/cleaned.bin`，配准模型保留在 `cache/aligned.bin`，卷帘校正阶段的所选模型在 `cache/shutter.bin`。程序更新后使用新工作目录。

`run.ps1` 增量编译后启动 `build/ninja/bin/geosunpre.exe`，启动日志和 `info` 显示程序路径、构建版本及时间。

## 模型与工作目录

默认模型目录为项目根目录下的 `models/`，TensorRT 固定为 10.13.3.9：

```text
models/
  manifest.json
  engines/{detector,descriptor,matcher,geometry,grounding}.engine
  runtime/nvinfer_10.dll
```

匹配器默认 FP16，Softmax、Reduce 和 Normalization 层保留 FP32；所有浮点输入输出保持 float32。原图按最长边 1536 缩放后送入动态检测器，横图、竖图及不同分辨率共用同一模型包。geometry 为 MoGe-2 ViT-L normal（FP16，644×644 透视子图），供默认的 MoGe 监督使用；grounding 为 SAM3（FP16，1008×1008 整幅照片），供掩码阶段区分天空、人物、影子与人形展品。模型准备后自动用真实照片验证动态宽高、五模型输出、动态描述子数量和匹配补零行为。输入范围、设备和精度参数见[模型文档](docs/models.md)。

模型目录中完整且兼容的包会校验后复用。更换 GPU 或匹配精度时使用新模型目录，并用 `--models` 指定；更换图像分辨率不需要重建。新生成的目录同时保留 ONNX 和准备元数据，模型资源不进入 Git。

工作目录保存冻结的输入、配置和模型位置。输入、配置或模型改变时使用新工作目录；续跑使用工作目录中记录的模型。不同请求的交付包拒绝覆盖。缓存格式与实现标记须匹配，详见[缓存约定](docs/cache.md)。

进度在 `state.json`，最终结果在 `result.json`，交付目录也附带一份 `result.json`。`cache/*_profile.json` 和 `cache/sfm.json` 保存前端耗时、BA 收敛、轨迹和几何质量统计，`result.json` 同时保存联合配准各项验收结果。

## 打包部署

交付包面向算力 ≥ 8.0 的 NVIDIA GPU。提交源码后运行：

```powershell
.\package.ps1
```

`package.ps1` 自动增量编译，并从本机 `models/` 的 ONNX 首次生成 `ampere_plus` 引擎，之后按本机模型清单复用缓存。也可以用 `-Models` 指定已准备好的兼容模型。通过五模型检查和文件哈希回读后，输出 `packages/GeosunPre-版本-提交-ampere_plus/`。可用 `-Output` 指定一个新目录，或用 `-Image` 指定用于验证的真实照片。

独立部署时，将整个包目录复制到目标 Windows 机器，按包内 README 运行。独立 EXE 默认读取同级 `models/`，详细说明见[部署文档](docs/deployment.md)。

预处理和训练作为整体交付时，训练配置以 `configs/trainer.json` 为准。提交后在同级 LFS 仓库执行 `scripts\package-windows-trainer.cmd`，它默认读取这份配置并记录当前 GeosunPre 提交；再组装整体包：

```powershell
.\package.ps1 -BundleVersion 1.1.0 -Preprocess packages/GeosunPre-版本-提交-ampere_plus -Trainer ..\LichtFeld-Studio\build-delivery\Geosun-3DGS-Trainer-sm80-时间
```

`-BundleVersion` 指定 Geosun 3DGS 的整包版本。正式版用 `1.1.0` 这类版本号，候选版用 `1.1.0-rc.2` 这类版本号。默认输出 `packages/Geosun-3DGS-1.1.0-Windows-x64.zip`、同名解压目录和 `.zip.sha256` 文件。整包与两个子包各自的版本都记录在 `bundle.json`，发出新包时分配新的整包版本。

`package.ps1` 在整包模式下编译启动器，原样复制两个子包到 `preprocess/` 和 `trainer/`。脚本核对文件哈希、提交和训练配置，并检查启动器、五模型和训练器。整体包用法见 [bundle/README.md](bundle/README.md)。

构建依赖使用固定版本；Ceres 使用轻量日志和 Eigen 稀疏求解，OpenCV 只构建图像处理所需模块。Ceres 回代补丁由 CMake 自动应用，清空 `build/` 后重新构建同样生效。
