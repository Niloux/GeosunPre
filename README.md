# GeosunPre

原生 C++17/CUDA 双鱼眼与 LAS 重建程序。单进程完成语义掩码、LoMa 特征与匹配、自有 SfM、共享鱼眼内参优化、LAS 配准、稀疏点清理、COLMAP/PLY 及 LAS 深度和法向监督导出，可选原图精修。运行不需要 Python、LibTorch、COLMAP、WSL 或联网。

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
| `run.ps1 run` | 使用上述程序和 `models/`，完成重建、LAS 配准、独立稀疏点清理及 COLMAP/PLY、LAS 深度和法向监督导出；原图精修默认关闭 |

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

默认交付 `WORK/colmap/`，包含原图、掩码、鱼眼 COLMAP 模型、5 cm LAS 初始化 PLY，以及 LAS 深度和法向监督图。`--no-ply` 省略 PLY，`--ply-voxel` 调整体素边长；`--no-las-supervision` 省略深度、法向图和 `supervision.json`。完整结构见[输出约定](docs/output.md)。

`run` 的导出阶段及独立 `export`、`refine`、`cleanup` 默认从原始有效 LAS 给已注册照片生成 Spirula 可读取的深度和法向图。已有工作目录可以直接导出到新目录，无需重跑重建：

```powershell
.\run.ps1 export --work "work/场景" --output "work/场景-supervised"
```

旧的 `--las-supervision` 显式开启参数仍可使用，与 `--no-las-supervision` 互斥。旧交付包保持原样；给旧结果补监督或改变导出选项时使用新交付目录。续跑已有无监督交付时，可指定新 `--output`，或加 `--no-las-supervision` 复用原包。

默认最长边 1600 px，深度为沿视线距离的 16 位毫米 PNG，法向使用 OpenCV 相机坐标并朝向相机。`--supervision-max-size` 调整最长边（128–4096），`--supervision-boundary` 设置屏蔽边界宽度（默认 6 个深度图像素，1–32）。只有静态区域内、具有可靠局部几何且非掠视角的像素参与监督；无效深度为 0，无效法向为黑色，原照片及 RGB 掩码保持不变。

监督导出只解析一次 LAS，并在相机批次之间复用有效整数坐标。`--supervision-cache-mb` 设置 GPU 坐标缓存上限，默认 1024 MiB（1–4096），且不超过开始时空闲显存的四分之一。超出预算时复用临时紧凑坐标文件，以双缓冲重叠上传和投影；内存保持有界，成功后清除临时文件。该选项不改变输出像素或重建缓存。

几何过滤不识别玻璃材质。可用 `--supervision-mask-dir DIR` 追加仅影响几何的排除掩码：按 `DIR/{L,R}/**.JPG.png` 保存原图尺寸灰度 PNG，仅 255 保留；未提供文件的照片沿用原静态区域。不同排除掩码使用新交付目录。

Spirula 直接读取交付包的 `depths/`、`normals/`。训练时显式设置 `input_depth_is_ray_depth=true` 并开启相应监督权重；现有深度相关性损失约束形状，不是绝对米制深度拟合。`supervision.json` 记录每图有效像素、生成参数及文件身份，`check` 会回读监督图。

在 Spirula 目录中可从较小权重开始试训，并与关闭监督的相同设置对照评估：

```powershell
.\build_vulkan\spirula.exe train 3dgs --data "交付目录" --primitive 3dgut --mask-mode segment_and_ignore --input-depth-is-ray-depth 1 --depth-supervision-weight 0.01 --normal-supervision-weight 0.002 --use-bilateral-grid-for-geometry 0
```

默认阶段为 `prepare → masks → features → matches → sfm → align → cleanup → export`。首次运行加 `--refine`，或在配置中设置 `"refine": true`，会在 `align` 与 `cleanup` 之间增加原图精修；默认 `false`。该选择随配置冻结，续跑自动沿用，改变选择须使用新工作目录。

`run --work WORK` 复用已完成阶段并恢复检查点；`run --work WORK --stage 阶段` 单独执行指定阶段，须先完成前置阶段。`--stage refine` 只用于已启用精修的工作目录。`prepare` 检查真实输入、相机和导航转换，不加载模型。

| 阶段 | 主要工作 | 主要产物 |
| --- | --- | --- |
| `prepare` 输入准备 | 绑定 L/R 照片与 CameraPos，核对尺寸、标定、LAS 和导航坐标转换；按 `.pro` 建立鱼眼内参，冻结输入及配置 | `scene.json`、`state.json` |
| `masks` 语义掩码 | 在虚拟透视视图中分割天空和人物，融合回鱼眼图，并限制镜头有效域；区分静态、天空与忽略区域 | `cache/masks/` |
| `features` 特征提取 | 在静态有效区域检测关键点，提取描述子、检索向量和原图颜色，保留空间覆盖 | `cache/features/` |
| `matches` 图像匹配 | 合并时间近邻、导航回访及外观检索候选，执行 LoMa 匹配，缓存对应关系；鱼眼几何验证在下一阶段执行 | `cache/pairs.json`、`cache/matches/` |
| `sfm` 视觉重建 | 鱼眼几何验证、建立轨迹、增量注册相机和三角化；BA 联合优化位姿、点和共享内参，补全轨迹，最后固定相机恢复未分配观测形成的有效点，并做几何验收 | `cache/visual.bin`、`cache/sfm.json` |
| `align` LAS 配准 | 导航位置提供尺度及初始坐标对齐，多尺度点到平面优化细化七参数相似变换；可靠表面支持时尝试联合约束相机与点，验收后采用 | `cache/aligned.bin`、`cache/alignment.json` |
| `refine` 可选原图精修 | 原始分辨率图块匹配及闭环检查，联合优化相机、共享内参和可靠轨迹；通过留出观测验证后采用，否则保留对齐模型 | `cache/refined.bin`、`cache/refinement.json`、`cache/refinement_result.json` |
| `cleanup` 稀疏点清理 | 对上一步所选模型独立清理、验收；精修关闭或候选被拒绝时仍正常执行，保留可靠双视图初始化点 | `cache/cleaned.bin`、`cache/cleanup.json`、`result.json` |
| `export` 交付 | 导出所选模型、已注册原图和掩码、鱼眼 COLMAP 稀疏模型，默认附带 LAS PLY、深度及法向监督；在临时目录回读校验通过后发布 | `colmap/` |

导航回访候选取 5 m 内、曝光时间相隔超过 10 s 的最近 20 幅照片，用于补充重复经过同一区域时的跨时段连接。它属于默认初始匹配，与停滞后的可选补匹配轮数不同。CameraPos 只提供位置；视觉姿态由 SfM 求解，不使用输入四元数，也不施加场景专用的固定角度补偿。

SfM 默认不自动补匹配；首次停滞时，对已注册部分完成优化、几何验收及 LAS 配准，通过后照常交付 COLMAP。配置 `sfm_repair_rounds` 可设为 `0`（默认）、`1` 或 `2`，表示停滞或照片支撑不足时最多补匹配的轮数，用完后处理已注册部分。交付仅包含已注册照片及其掩码，`manifest.json` 和 `quality.json` 的 `registration` 保存注册数量与未注册照片名单；`complete=false` 表示部分注册。验收不通过仍明确报错。

默认配置唯一来源是 [configs/default.json](configs/default.json)，构建时编译进程序，独立运行包无需携带。`--config` 文件只需列出要修改的键，其余沿用默认值；未知键报错。内参标定默认使用全部有效轨迹（`calibration_tracks=all`），随后锁定内参补全轨迹并精调。标定选项及坐标约定见[输入文档](docs/input.md)。

`workers` 默认 4，控制前端及增量注册阶段 BA 的并行数；SfM 几何验证逐对独立且结果与线程数无关，使用全部硬件线程（不少于 `workers`，最多 32），实际线程数记录在 `cache/sfm.json` 的 `geometry_threads`；`ba_workers` 默认 8，单独控制注册结束后的标定与全局精调、LAS 联合优化和已启用的原图精修 BA，两者均可设为 1–32。求解器、双精度残差、收敛精度和验收门限保持原有设置；独立 `refine` 的 `--workers` 仍控制其 BA。`cache/sfm.json` 的每次优化记录包含构建、预处理、残差、导数、线性求解耗时及逐轮迭代信息。

原图精修默认关闭。开启后保留对齐模型，相机候选未通过时使用原模型，原因保存在 `cache/refinement.json`。后续 `cleanup` 总是独立执行；清理候选未通过时保留清理前的模型，原因保存在 `cache/cleanup.json`。读图、求解或文件错误仍明确报错。

cleanup 主要去除视觉稀疏模型中重投影不一致、视差不足、深度不稳定或缺少有效 LAS 支撑的点；它不执行原图图块匹配、不优化相机。可靠 LAS 平面支撑不足时保留输入模型，并在清理报告中记录原因。

新场景需要原图精修时，在首次运行启用：

```powershell
.\run.ps1 run --input "D:/数据/场景" --work "work/场景-精修" --refine
```

已有工作目录可用独立命令处理，结果及新交付写入新目录。独立 `refine` 仅精修相机与点，独立 `cleanup` 仅清理稀疏结构；两者均不重跑前端和 SfM。新目录先在同级 `.pending` 中完整生成后再发布，处理期间锁定来源目录；`--no-ply`、`--ply-voxel` 与 `export` 含义相同：

```powershell
.\run.ps1 refine --work "work/场景" --output "work/场景-精修" --workers 8
.\run.ps1 cleanup --work "work/场景" --output "work/场景-清理" --workers 8
```

精修在原始分辨率上对齐邻近视图的图块，使用鱼眼投影和可靠 LAS 平面处理图块变形，以亚像素位置和亮度参数联合精调，并检查双向往返与轨迹闭环一致性。匹配图块做轻度抗噪处理，交付图像仍保留原图。

随后以已有重建为基准联合优化共享内参、相机和点：已精修观测使用原图对应，其他训练观测约束相对原投影的变化，共用 2 px Huber 损失；LAS 法向约束相对原点的位移，保留导航先验。这避免重复拟合未精修的粗特征噪声，也不将已通过配准的点再次强拉到平面零残差。至少三视图的轨迹提供相机支撑；两视图点不参与共享内参与相机的精修。

点编号满足 `%5 == 0` 的轨迹固定留出，并按逐图覆盖补充整条轨迹，不参加相机优化。成功精修的留出轨迹须同时改善误差中位数和 P90；全部留出轨迹、每幅照片及有足够样本的格子还须通过退化检查。另外检查完整模型的几何质量、照片支撑及同一组 LAS 平面残差。留出误差反映内部一致性，不能代替独立精度检查；误差验收始终使用实际观测和绝对 LAS 平面距离，强力删点不会用于帮助相机候选通过验收。

稀疏点清理作为单独候选：至少两次观测、视差至少 1.5°，单观测误差不超过 `min(3 px, quality_p90_px)`。距离和点位不确定度门限按 `las_median_m` 及 LAS 量化精度计算，默认分别为 10 cm 和 5 cm；该不确定度是固定相机的局部估计，不代表独立精度。清理后支撑不足的照片会从候选移除，但只有总照片保留至少 99%、每个镜头保留至少 98%、原可靠 LAS 支撑轨迹保留至少 90%，且图像图仍连通、剩余照片达到 32 个观测和 8 个覆盖格子、几何质量通过，才采用清理。否则保留清理前的模型，不强行删点。清理不修改相机参数，也不改变由原始 LAS 导出的初始化 PLY。

默认流程的最终所选模型在 `cache/cleaned.bin`，配准模型保留在 `cache/aligned.bin`；开启精修时还保留 `cache/refined.bin`。独立处理的新交付在新目录的 `colmap/`；独立 `refine` 的所选模型为 `refined.bin`，独立 `cleanup` 为 `cleaned.bin`。这些目录可用 `export` 再导出，不作为 `run` 的阶段恢复目录。当前实现标记为 16，旧版工作目录、交付及训练结果保持原样，仍可独立导出或处理；新流程使用新工作目录。

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

交付包面向任意算力 ≥ 8.0 的 NVIDIA GPU：程序编译 sm_80/86/89/120 机器码及 compute_80 PTX，模型使用 `ampere_plus` 引擎。先准备通用模型，提交源码后编译并打包：

```powershell
.\prepare-models.ps1 -Models models-ampere-plus -OnnxFrom packages/onnx-bundle-v1 -HardwareCompatibility ampere_plus
.\build.ps1
.\package.ps1
```

`package.ps1` 默认读取 `models-ampere-plus/`，拒绝含未提交改动的程序，输出 `packages/GeosunPre-版本-提交-引擎级别/` 及同名 `.zip`、`.zip.sha256`。包内含程序、运行库、模型引擎与 TensorRT 运行库、用户文档（输入、输出、缓存）、许可证和记录全部文件 SHA256 的 `package.json`。打包后用包内程序执行四模型检查并回读文件哈希，通过后发布。支持 `-Models`、`-Output`、`-Image`；输出目录须不存在。

将 zip 复制到目标 Windows 机器解压，按包内 README 运行。独立 EXE 默认读取同级 `models/`，详细说明见[部署文档](docs/deployment.md)。

构建依赖使用固定版本；Ceres 使用轻量日志和 Eigen 稀疏求解，OpenCV 只构建图像处理所需模块。Ceres 回代补丁由 CMake 自动应用，清空 `build/` 后重新构建同样生效。
