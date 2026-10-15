# GeosunPre

原生 C++17/CUDA 双鱼眼与 LAS 重建程序。单进程完成语义掩码、LoMa 特征与匹配、自有 SfM、共享鱼眼内参优化、LAS 配准、原图精修和 COLMAP/PLY 导出。运行不需要 Python、LibTorch、COLMAP、WSL 或联网。

## 开始使用

在 Windows x64 上安装 Visual Studio 2022 C++ 桌面工具链及 CMake 工具、Windows SDK、CUDA Toolkit 12.8、CMake 3.24 以上，然后在项目根目录的普通 PowerShell 中执行：

```powershell
.\build.ps1
.\prepare-models.ps1
.\run.ps1 run --input "D:/数据/场景" --work "work/场景"
```

这就是完整的默认流程。构建始终使用当前源码；GPU 预处理、特征颜色缓存、鱼眼 PnP、解析 BA 导数和迭代 Schur 优化均默认启用。

| 步骤 | 默认行为 |
| --- | --- |
| `build.ps1` | 使用 Ninja 编译 Release 程序到 `build/ninja/bin/geosunpre.exe`，自动应用 Ceres Schur 回代修复 |
| `prepare-models.ps1` | 为 GPU 0 准备 `models/`，匹配器默认 FP16，检测器支持动态宽高 |
| `run.ps1 run` | 使用上述程序和 `models/`，完成重建、LAS 配准、原图精修及 COLMAP/PLY 导出 |

构建统一使用单配置 Ninja Release。脚本自动加载 x64 MSVC 环境，优先使用 PATH 中的 Ninja，否则使用 Visual Studio 自带的 Ninja。首次构建下载固定依赖到 `build/ninja/_deps/` 并编译，此后复用缓存。日常构建只编译变化影响到的文件，CMake 文件变化由 Ninja 自动重新配置；修改 preset 后自动重新配置，也可用 `build.ps1 -Configure` 显式重新配置，`-Jobs` 控制并行数（默认 8）。构建版本与时间单独生成，不会使全部源码因时间变化而重编。

首次准备模型下载转换工具和固定权重到 `.cache/model-prepare/`，需要联网及数 GB 下载空间。此后复用缓存。模型准备使用独立 Python 环境，运行程序与独立运行包使用 C++/CUDA/TensorRT。

## 日常运行

```powershell
.\run.ps1 run --work "work/场景"
.\run.ps1 prepare --input "D:/数据/场景" --work "work/场景"
.\run.ps1 export --work "work/场景" --output "D:/成果/场景"
.\run.ps1 check --output "D:/成果/场景"
.\run.ps1 info
```

默认交付 `WORK/colmap/`，包含原图、掩码、鱼眼 COLMAP 模型和 5 cm LAS 初始化 PLY。`--no-ply` 省略 PLY，`--ply-voxel` 调整体素边长。完整结构见[输出约定](docs/output.md)。

阶段为 `prepare → masks → features → matches → sfm → align → refine → export`。`run --work WORK` 复用已完成阶段并恢复检查点；`run --work WORK --stage 阶段` 单独执行指定阶段，须先完成前置阶段。`prepare` 检查真实输入、相机和导航转换，不加载模型。

| 阶段 | 主要工作 | 主要产物 |
| --- | --- | --- |
| `prepare` 输入准备 | 绑定 L/R 照片与 CameraPos，核对尺寸、标定、LAS 和导航坐标转换；按 `.pro` 建立鱼眼内参，冻结输入及配置 | `scene.json`、`state.json` |
| `masks` 语义掩码 | 在虚拟透视视图中分割天空和人物，融合回鱼眼图，并限制镜头有效域；区分静态、天空与忽略区域 | `cache/masks/` |
| `features` 特征提取 | 在静态有效区域检测关键点，提取描述子、检索向量和原图颜色，保留空间覆盖 | `cache/features/` |
| `matches` 图像匹配 | 合并时间近邻、导航回访及外观检索候选，执行 LoMa 匹配，缓存对应关系；鱼眼几何验证在下一阶段执行 | `cache/pairs.json`、`cache/matches/` |
| `sfm` 视觉重建 | 鱼眼几何验证、建立轨迹、增量注册相机和三角化；BA 联合优化位姿、点和共享内参，补全轨迹，最后固定相机恢复未分配观测形成的有效点，并做几何验收 | `cache/visual.bin`、`cache/sfm.json` |
| `align` LAS 配准 | 导航位置提供尺度及初始坐标对齐，多尺度点到平面优化细化七参数相似变换；可靠表面支持时尝试联合约束相机与点，验收后采用 | `cache/aligned.bin`、`cache/alignment.json` |
| `refine` 原图精修 | 原始分辨率图块匹配及闭环检查，联合优化相机、共享内参和可靠轨迹；留出观测验证后再单独验收点云清理，保留可靠双视图初始化点 | `cache/refined.bin`、`cache/refinement.json`、`result.json` |
| `export` 交付 | 导出所选模型、已注册原图和掩码、鱼眼 COLMAP 稀疏模型及可选 LAS PLY；在临时目录回读校验通过后发布 | `colmap/` |

导航回访候选取 5 m 内、曝光时间相隔超过 10 s 的最近 20 幅照片，用于补充重复经过同一区域时的跨时段连接。它属于默认初始匹配，与停滞后的可选补匹配轮数不同。CameraPos 只提供位置；视觉姿态由 SfM 求解，不使用输入四元数，也不施加场景专用的固定角度补偿。

SfM 默认不自动补匹配；首次停滞时，对已注册部分完成优化、几何验收及 LAS 配准，通过后照常交付 COLMAP。配置 `sfm_repair_rounds` 可设为 `0`（默认）、`1` 或 `2`，表示停滞或照片支撑不足时最多补匹配的轮数，用完后处理已注册部分。交付仅包含已注册照片及其掩码，`manifest.json` 和 `quality.json` 的 `registration` 保存注册数量与未注册照片名单；`complete=false` 表示部分注册。验收不通过仍明确报错。

默认配置见 [configs/default.json](configs/default.json)，可通过 `--config` 指定。内参标定默认使用全部有效轨迹（`calibration_tracks=all`），随后锁定内参补全轨迹并精调。标定选项及坐标约定见[输入文档](docs/input.md)。

原图精修默认在 LAS 对齐后自动执行，不需要另跑命令。它保留对齐模型，分别验收相机优化和稀疏点清理；相机候选未通过时使用原模型，清理候选未通过时使用精修后未清理的模型。未采用原因保存在 `cache/refinement.json`，读图、求解或文件错误仍明确报错。可以用独立命令升级已有工作目录，或在新目录继续精修：

```powershell
.\run.ps1 refine --work "work/场景" --output "work/场景-精修" --workers 8
```

精修在原始分辨率上对齐邻近视图的图块，使用鱼眼投影和可靠 LAS 平面处理图块变形，以亚像素位置和亮度参数联合精调，并检查双向往返与轨迹闭环一致性。匹配图块做轻度抗噪处理，交付图像仍保留原图。

随后以已有重建为基准联合优化共享内参、相机和点：已精修观测使用原图对应，其他训练观测约束相对原投影的变化，共用 2 px Huber 损失；LAS 法向约束相对原点的位移，保留导航先验。这避免重复拟合未精修的粗特征噪声，也不将已通过配准的点再次强拉到平面零残差。至少三视图的轨迹提供相机支撑；两视图点不参与共享内参与相机的精修。

点编号满足 `%5 == 0` 的轨迹固定留出，并按逐图覆盖补充整条轨迹，不参加相机优化。成功精修的留出轨迹须同时改善误差中位数和 P90；全部留出轨迹、每幅照片及有足够样本的格子还须通过退化检查。另外检查完整模型的几何质量、照片支撑及同一组 LAS 平面残差。留出误差反映内部一致性，不能代替独立精度检查；误差验收始终使用实际观测和绝对 LAS 平面距离，强力删点不会用于帮助相机候选通过验收。

稀疏点清理作为单独候选：至少两次观测、视差至少 1.5°，单观测误差不超过 `min(3 px, quality_p90_px)`。距离和点位不确定度门限按 `las_median_m` 及 LAS 量化精度计算，默认分别为 10 cm 和 5 cm；该不确定度是固定相机的局部估计，不代表独立精度。清理后支撑不足的照片会从候选移除，但只有总照片保留至少 99%、每个镜头保留至少 98%、原可靠 LAS 支撑轨迹保留至少 90%，且图像图仍连通、剩余照片达到 32 个观测和 8 个覆盖格子、几何质量通过，才采用清理。否则保留相机精修结果，不强行删点。

最终所选模型在 `cache/refined.bin`，精修前模型在 `cache/aligned.bin`。独立精修的新交付在新目录的 `colmap/`；原始输入、工作目录及已有训练结果保持原样。独立精修目录可用 `export` 再导出，不作为 `run` 的阶段恢复目录。当前实现标记为 14，旧版工作目录继续保留；使用默认新流程时另建工作目录。

`run.ps1` 固定启动 `build/ninja/bin/geosunpre.exe`，启动日志和 `info` 显示程序路径、构建版本及时间。重新编译只需执行 `build.ps1`。

## 模型与工作目录

默认模型目录为项目根目录下的 `models/`，TensorRT 固定为 10.13.3.9：

```text
models/
  manifest.json
  engines/{detector,descriptor,matcher,segmentation}.engine
  runtime/nvinfer_10.dll
```

匹配器默认 FP16，Softmax、Reduce 和 Normalization 层保留 FP32；所有浮点输入输出保持 float32。原图按最长边 1536 缩放后送入动态检测器，横图、竖图及不同分辨率共用同一模型包。模型准备后自动用真实照片验证动态宽高、四模型输出、动态描述子数量和匹配补零行为。输入范围、设备和精度参数见[模型文档](docs/models.md)。

模型目录中完整且兼容的包会校验后复用。更换 GPU 或匹配精度时使用新模型目录，并用 `--models` 指定；更换图像分辨率不需要重建。已有固定尺寸包可执行 `.\prepare-models.ps1 -Models models/dynamic -OnnxFrom models`，只重建检测器，然后使用 `--models models/dynamic`。新生成的目录同时保留 ONNX 和准备元数据，模型资源不进入 Git。

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
