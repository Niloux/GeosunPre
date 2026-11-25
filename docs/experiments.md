# 实验记录

统一记录 GeosunPre 关键算法、配置和产物规格改动后的实测对照数据与取舍结论，便于跨对话延续上下文；只写已实际观测到的数据。

## 2026-11-22：小场景异步双鱼眼卷帘物理校验、等时长先验与走廊 Rig 约束修复

- 背景与诊断：在 `work/1122/小场景`（296 幅 `ShareUAV CM2000-2.55D` 3600×3600 异步双鱼眼，L/R 约 56 ms 交替触发）及下游 3DGS（30k + `--eval-all`）中，地砖等近景表面出现双边/模糊（如 `00071-L`）且走廊窗户及地面局部大幅错位（如 `00072-L`）。定位为两项叠加因素：
  1. 原 `ReadoutConsistency` 用同镜头约 1.6 s 相邻曝光视觉姿态差分近似瞬时角速度，在手持步行中将高频手腕抖动平均掉（`L` 解释率仅 `0.024 < 0.10`），误拒了实际有效（留出重投影中位数 `1.161 -> 0.933 px`）的卷帘模型，导致交付 `shutter.used=false`（下游 3DGS `0 / 296` 相机启用卷帘渲染）。
  2. 启用 IMU 物理校验后，BA 中 `L`/`R` 读出时间存在约 `-13 ms` 共模漂移（`readout_s = [-0.0468, +0.0223]`），导致 `polish` 时 `L` 触及 `-0.05 s` 边界失败回退；同时弱纹理走廊段 `72..78L` 在 `aligned.bin` 已有沿行进方向 `-6.3 cm` 的姿态漂移，而原有异步 `RigLink`（`σ=3 mm` + `HuberLoss(2)`）在 `6 mm` 以上即线性截断，未能将其拉回 rig 基线。
- 代码改动：
  - `src/sfm/shutter.cpp`：`ReadoutConsistency` 在 `imu_attitude.verified` 时优先使用曝光时刻 200 Hz IMU 角速度（无 IMU 时回退相邻曝光差分）并记录 `rate_source`；提取共享检查函数 `StructuralChecks`，新增在双轴卷帘均未通过时触发的全局快门 rig 回退 `GlobalRigFallback`（`ReanchorRigOutliers` + 全量轨迹/LAS 平面/二次 `RigLink` 全局快门 BA + 稳定性与平面门限）。
  - `src/sfm/optimization.h` / `src/sfm/optimization.cpp`：对同像素尺寸双镜头施加等时长读出先验 `EqualReadoutDuration`（`EqualReadoutSigma = 0.0002 s` 约束 `|τ_L| - |τ_R|`）；将异步 `RigLink` 位置先验 `AsyncRigPositionSigma` 从 `0.003 m` 收紧至 `0.001 m` 并统一取消 `RigLink` Huber 截断（采用二次残差）。
  - `src/sfm/shutter.h` / `src/app/pipeline.cpp`：补充 `ShutterCorrection::rig_refined`，使全局快门 rig 回退通过后输出修正模型至 `shutter.bin` 并保持 `shutter.used=false`（不附带 `rolling_shutter.json`）。

| 验证组 (`小场景` 30k `--eval-all`) | `shutter.used` | `readout_s` (s) | `polish` | Rig 残差 med/p90/max (cm) | 72L Rig 残差 (cm) | 3DGS PSNR / SSIM / LPIPS | 深度中位误差 (mm) | 法线误差 (°) | `00071-L` / `00072-L` PSNR | 分块行梯度 med (L / R eval px) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 基线复现 (`小场景-rs-base`) | false | — | — | — | — | 23.5275 / 0.8046 / 0.2534 | 8.2895 | 10.7159 | 23.10 / 17.52 | 2.034 / 1.263 |
| 仅 IMU 物理校验 (`小场景-rs-imu`) | true (296/296) | [-0.0468, +0.0223] | false | 0.51 / 3.07 / 6.28 | 6.28 | 24.2738 / 0.8269 / 0.2444 | 7.8899 | 10.6630 | 23.39 / 18.02 | 1.036 / 0.633 |
| 完整修复从零重建 (`小场景-fixed`) | true (296/296) | [-0.04232, +0.03851] | true | 0.33 / 0.60 / 0.85 | 0.31 | 24.3954 / 0.8314 / 0.2407 | 6.2423 | 10.6135 | 25.06 / 24.02 | 0.942 / 0.486 |

- `小场景-fixed` 附加指标：`validation_px` `1.1560 -> 0.9412`，`explained_rotation` `[0.434, 0.647]`，LAS 平面距离中位 `8.16 -> 3.86 mm` / P90 `29.31 -> 14.66 mm`，COLMAP 重投影中位 `0.7279 px` / P90 `1.7694 px`，`geosunpre check` 通过（`296 / 296` 注册）；3DGS 相对基线 239/296 视角提升，均值 `+0.868 dB`（`00072-L` `+6.50 dB`、`00073-L` `+6.23 dB`，`L71` 行梯度由 `2.08` 降至 `0.37 eval px`、`L72` 全局偏移由 `2.61` 降至 `0.07 eval px`）。
- 同步场景回归（`展厅-verify`，`--no-las-supervision` + `check`）：`588 / 588` 注册通过，COLMAP 重投影中位 `0.8890 px` / P90 `2.0693 px`（基线 `0.8898 / 2.0707 px`），`shutter.used=true`、`readout_s = [+0.02453, +0.02369]`（两镜头读出差从 `1.41 ms` 缩至 `0.84 ms`）、`polish.used=true`、`validation = 1.4325 -> 1.1537 px`，无退化。
- 边界与结论：采用。`GlobalRigFallback` 的真实生效仅发生在双轴卷帘均被拒绝且非强卷帘失真的场景（在 `小场景` 临时关闭卷帘验收测试回退路径时，因真实图像含强卷帘畸变导致全局快门 BA 违反 `per_image_stability` 而被门限安全拒绝），待后续真实全局快门或低动态场景覆盖实际命中场景。

## 2026-11-22：200 Hz IMU 相对旋转先验前移进 SfM（手眼自标定 + 已知旋转验证/PnP + BA 先验）

- **核心改动**：借鉴 Spirula `PriorSource` / `KnownRotation` 思路，在保持全局快门 SfM 不绑 L/R 刚性 rig 的前提下，将 `ReadExposureKinematics` 提供的 200 Hz POS 姿态前移进 SfM 全链路：
  1. [optimization.h](../src/sfm/optimization.h) / [optimization.cpp](../src/sfm/optimization.cpp)：新增 `RelativeRotationCost` 与 `Optimize(..., relative_rotations)`，在非卷帘 BA 中对同镜头连续曝光（`Δt <= 5 s`）施加相对旋转先验（`ImuRelativeRotationSigma = 0.5°`，`2σ Huber`）。
  2. [reconstruction.cpp](../src/sfm/reconstruction.cpp)：新增 `ImuPriorContext`（Pass 1 五点初验后各镜头独立跑 3 轮 `A X = X B` IRLS 手眼自标定；转入导航坐标系后再用注册姿态稳健四元数均值刷新视轴外参）、`VerifyPairs` Pass 2 已知旋转 2 点 RANSAC + 5-DOF 近旋转 LM 精调（救回漏配对并改判/剔除偏离先验的错配）、`Register` 已知旋转 2 点 PnP 回退与离群复核（`SolvePnPKnownRotation`），并将统计记录写入 `cache/sfm.json` 的 `imu_prior`。

### 实景对比：`小场景`（296 幅异步）与 `展厅`（588 幅同步）

两套实景全流程输出至独立目录（`work/小场景-imu-sfm-out` 与 `work/展厅-imu-sfm-out`）并保留在本地，均通过 `run.ps1 check` 校验（`passed: true`）。`小场景` 基线取同代码世代（含 `bdda234` 卷帘修复）的 `work\1122\小场景-fixed`；`展厅` 基线取 `work\1122\展厅`。

| 指标 | `小场景` 基线 (`1122/小场景-fixed`) | `小场景` IMU-SfM (`小场景-imu-sfm-out`) | `展厅` 基线 (`1122/展厅`) | `展厅` IMU-SfM (`展厅-imu-sfm-out`) |
| --- | :---: | :---: | :---: | :---: |
| 注册情况（最终交付） | `296 / 296` | `296 / 296`（`pnp_rescued=2`） | `588 / 588`（`rig_rescue=3`） | `588 / 588`（`rig_rescue=3`） |
| 手眼 / 终态视轴中位残差 (L / R) | — | `0.394° / 0.188°` → `0.580° / 0.188°` | — | `0.607° / 0.647°` → `0.607° / 0.421°` |
| Pass 2 核验（check/rescue/overrule/reject） | — | `782 / 8 / 199 / 151` | — | `2066 / 24 / 566 / 99` |
| 验证配对数 / 几何耗时 | `3,258` / `3.64 s` | `3,115` / `4.03 s` | `7,958` / `6.42 s` | `7,883` / `7.52 s` |
| `sfm` 总耗时 | `67.17 s` | **`58.84 s`**（`-12.4%`） | `127.60 s` | `141.61 s`（`+11.0%`） |
| 卷帘前点数 / 观测数 (`source_quality`) | `161,364` / `534,420` | **`166,230` / `548,115`**（`+3.0% / +2.6%`） | `206,648` / `958,662` | **`208,565` / `972,749`**（`+0.93% / +1.47%`） |
| 交付点数 / 观测数 (`quality`) | `118,837` / `404,029` | **`122,614` / `413,944`**（`+3.2% / +2.5%`） | `152,915` / `738,818` | **`154,227` / `748,355`**（`+0.86% / +1.29%`） |
| 交付长轨迹 (`tracks >= 3 / >= 8`) | `56,061` / `6,527` | **`57,300` / `6,675`** | `71,183` / `17,364` | **`71,891` / `17,630`** |
| 交付重投影 `median / p90` | `0.7279 / 1.7694 px` | **`0.7247 / 1.7625 px`** | `0.8898 / 2.0707 px` | `0.8940 / 2.0763 px` |
| 导航位置残差 `median / p90` | `0.0432 / 0.0521 m` | **`0.0421 / 0.0508 m`** | `0.0213 / 0.0367 m` | **`0.0211 / 0.0357 m`** |
| LAS 点到面 `median / p90` | `0.00747 / 0.02828 m` | **`0.00717 / 0.02703 m`** | `0.00563 / 0.01644 m` | `0.00583 / 0.01655 m` |
| 卷帘 `used` / `readout_s` | `true` / `[-0.04232, +0.03851]` | `true` / `[-0.04192, +0.03862]` | `true` / `[+0.02439, +0.02298]` | `true` / `[+0.02449, +0.02359]` |
| 卷帘留出 `before -> after` 中位 | `1.1560 -> 0.9412 px` | `1.1650 -> 0.9534 px` | `1.4319 -> 1.1525 px` | **`1.4277 -> 1.1506 px`** |

- **结论**：双场景手眼自标定稳定收敛（`0.19°–0.65°`），已知旋转二次核验精准清理了误匹配并救回 `8` / `24` 对弱纹理配对，增量注册在 `小场景` 直接救回 `2` 帧 AP3P 失败帧，双场景长轨迹与总有效观测数均稳定提升（`+1.5%–2.6%`），导航位置与重投影一致性保持或微幅改善，且完全不干扰后续 `shutter` 卷帘读出时间求解（`shutter.used=true` 稳定保留）。`展厅` 的 `00214–00216-R` 白墙帧因匹配候选中不足 24 条共视观测仍由 `shutter.rig_rescue` 完整补回。确立保留为主线默认，无需新增 CLI 开关。

## 2026-11-23：预处理导出阶段自适应生成 `trainer.json`（LFS 零改动）

- **问题与设计**：此前下游固定共用同一份 [configs/trainer.json](../configs/trainer.json)（`iterations = 30000`、`max_cap = 3000000`），未随图像规模（数百至数千张）、ROI 几何体素复杂度及训练机显存自适应调整。在**完全不修改 `LichtFeld-Studio`** 的前提下，在 `GeosunPre` 侧完成全链路自适应配置生成：
  1. [CMakeLists.txt](../CMakeLists.txt) 将 `configs/trainer.json` 内嵌编译为 `DefaultTrainerConfig`；新增 [trainer_config.h](../src/io/trainer_config.h) / [trainer_config.cpp](../src/io/trainer_config.cpp)，由 `ExportBundle` 在导出期自动生成 `<output>/trainer.json`（同步写入 `manifest.json` 的 `trainer_config` 摘要并在 `check` 中校验；存量交付包可用 `geosunpre trainer-config --output DIR [--vram-gb GB] [--trainer-config JSON]` 补生成）。
  2. **迭代步数与调度同步缩放**：按 `N_train`（已注册图像数扣除 `test_every` 留出比例）以 `550` 张为锚点幂律推导 `iterations = round_1000(clamp(base_iters * (N_train / 550)^0.6, 20000, 120000))`，并按同一比例缩放 `start_refine`、`stop_refine`、`reset_every`、`refine_every`、`morton_reorder_interval`、`sh_degree_interval`、`grow_until_iter`、`color_variance_warmup_iters`、`eval_steps` 与 `save_steps`，保持 `steps_scaler = 1.0` 避免重复缩放。
  3. **高斯上限与初始化点云协同**：取几何需求 `cap_geom = 3.0 * geom_points`（优先取 `roi_voxel_points_before_cap`）与视角支撑 `cap_views = 8000 * N_train` 较小值，受显存硬上限 `cap_vram = round_100k(clamp((vram_gb - 2.5) * 1800000, 4000000, 24000000))`（`8 GB -> 9.9M`、`24 GB -> 24M`）钳位并下限保护 `1.5M`，写出 `max_cap`（LFS 自动按 `grow_fraction = -1` / `shs_lr = -1` 联动）；`ExportLasPly`（[las.cpp](../src/io/las.cpp)）同步记录 `roi_voxel_points_before_cap` 并将 PLY 封顶联动为 `min(2500000, max(100000, round(0.75 * max_cap)))`。
  4. [bundle/main.cpp](../bundle/main.cpp) 调整为训练优先加载 `<delivery>/trainer.json`，不存在时回退包内默认模板；[output.md](output.md) 已同步更新契约。

### 三场景实景验证（`RTX 4090`，`23.99 GiB` 显存，`--eval-all`，历史对照基线）

| 场景 / 方案 | `N_reg` / `N_train` | `geom_points` (`roi_voxel`) | `iterations` / `max_cap` / `ply_cap` | 训练耗时 | 裁切后 `#GS` | 采样显存 (`nvidia-smi`) | `Mask PSNR / SSIM / LPIPS` | `Aligned PSNR` | 法线角 (°) ↓ | 深度 med / p90 (mm) ↓ |
| --- | :---: | ---: | :---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `小场景` 固定上限对照 | `296` / `296` | `~845k` | `30,000 / 3,000,000 / 2,500,000` | `877.3 s` | `2,994,590` | — | `24.4838 / 0.8286 / 0.2452` | `25.8417` | `7.6668°` | `6.17 / 33.12` |
| `小场景` 自适应 | `296` / `259` | `845,322` | `20,000 / 2,100,000 / 1,575,000` | **`492.8 s`**（`-43.8%`） | `2,096,943` | `6,068 MiB`（`11,400` 步采样） | `23.8773 / 0.8122 / 0.2734` | `24.9136` | `7.9020°` | `5.9110 / 30.0103` |
| `展厅` 固定上限对照 | `588` / `588` | `~882k` | `30,000 / 3,000,000 / 2,500,000` | `899.1 s` | `2,988,447` | — | `26.1516 / 0.8486 / 0.2717` | `26.8578` | `7.6643°` | `8.1261 / 73.0369` |
| `展厅` 自适应 | `588` / `514` | `881,016` | `29,000 / 2,600,000 / 1,950,000` | **`863.9 s`**（`-3.9%`） | `2,585,606` | `9,084 MiB`（`17,300` 步采样） | `26.1391 / 0.8474 / 0.2763` | `26.8017` | `7.6382°` | `8.3586 / 64.7211` |
| `篮球场` 自适应 | `536` / `469` | `5,143,902` | `27,000 / 3,800,000 / 2,500,000` | `835.0 s` | `3,758,049` | `8,666 MiB`（`17,600` 步采样） | `19.1562 / 0.5861 / 0.3393` | `19.3709` | `13.5641°` | `9.6396 / 113.1315` |

注：该自适应机制已在后续回归原版 LFS 默认参数后完整保留（模板切换见末节）。未验证项：未在真实 `8 GB` 物理显卡或数千张图数据集上做端到端训练压测。

## 2026-11-23：3DGUT 反向核性能修复（`LichtFeld-Studio`）

- **排查定位（基于 `work/小场景` `3000` 步分析与 `cuobjdump`）**：`RasterizeToPixelsFromWorld3DGSBwd.cu` 此前无条件为 `s_cv_target` / `s_cv_buf` 分配 `1,280 B` 共享内存（使 `CDIM=8` 达到 `10,240 / 10,496 B`）并保留内层分支，且存在 `pixel_weight` 时每像素重复读取 `*color_variance.pixel_weight_sum`；同时在 `pipelined_depth_own_size_ == true` 且先验已适配时存在重复 `fit_depth_prior` 调用。
- **保留改动**：
  - [RasterizeToPixelsFromWorld3DGSBwd.cu](file:///d:/Dev/LichtFeld-Studio/src/training/rasterization/gsplat/RasterizeToPixelsFromWorld3DGSBwd.cu)：为 `rasterize_to_pixels_from_world_3dgs_bwd_splat_kernel` 增加 `kColorVariance` 编译期模板参数（关闭时将 `s_cv_target`/`s_cv_buf` 缩至 1 元素，共享内存降回 `8960 / 9216 B`，`LOCAL:0`），并在开启分支前由 warp `lane 0` 读取 `*color_variance.pixel_weight_sum` 后 `__shfl_sync` 广播。
  - [trainer.cpp](file:///d:/Dev/LichtFeld-Studio/src/training/trainer.cpp)：在分辨率匹配且无需重拟合时复用已适配的深度先验并 `sync_to_stream`，避免重复调用 `fit_depth_prior`。
- **当时基准实测**（`work/小场景`，`max_cap = 2,100,000`、`iterations = 3,000`）：修复前 `49.094 s (61.1 iter/s)` → 修复后（关闭 color variance，warmup 期间）提速至 `45.170 s (66.4 iter/s)`（`-8.0%`）；开启 color variance 时为 `46.912 s (63.9 iter/s)`。该核优化在后续清理魔改正则参数时完整保留。

## 2026-11-23：回归原版 LFS 默认训练参数并切换至纯 MoGe-2 SSI 监督

- **背景与决策**：实测原版 LichtFeld Studio 默认参数训练效果优于此前叠加的魔改正则，确立两项同步重构：
  1. **对齐原版 LFS 默认参数并清理魔改代码**：以原版 LFS 默认配置为模板更新 [configs/trainer.json](../configs/trainer.json)（保留 `roi.json` 关联的 `crop_to_cropbox: true`、`cropbox_growth_inside: 0.7`，保持 `headless: true` 与 `depth_loss_mode: "ssi"`），并在 `LichtFeld-Studio` 中彻底删除魔改参数与实现（`free_space_*`、`depth_distortion_weight`、`normal_variance_weight`、`normal_prior_depth_weight`、`normal_prior_decay_*`、`depth_loss_gradient_weight`、`depth_loss_mode="metric"` / `metric_depth_anchor` / `DepthPriorError`），完整保留训练速度优化（per-splat GUT backward 及 `kColorVariance` 模板优化、8×8 sub-tile、SH 流、host sync 移除、mask 剔除、两阶段排序、bilagrid gather、`raw_cache_memory_ratio`、nvImageCodec 线程上限等）与移植功能（GUT 正则、`rolling_shutter.json`、`roi.json`、`LFS_TRAINER_ONLY`、`splat_mesh`/OSGB、`SegmentAndIgnore` 修复）。
  2. **监督切换为纯 MoGe-2 SSI**：`GeosunPre` 侧保留 `static_masks` 所需的单遍 LAS 最近深度面元投影，废弃原有 LAS 清洗深度 + 平面法向混合监督与 `--supervision-source` 开关，改为输出纯 MoGe-2 跨透视子图尺度对齐的 16 位 SSI 射线距离图（`uint16_png_ssi`，`[0.02, 1.0] * 65535`，`0` 无效）与 OpenCV 相机系朝向相机的密集法向图（`moge2_ssi` / `moge2_normals`）；`LichtFeld-Studio` 侧 `las_prior_sidecar.cpp` 自动关闭自动生成、保持 `depth_loss_mode = "ssi"` 并置 `normal_loss_space = camera-opencv`。
- **已观测验证**：
  - `GeosunPre` `tools/windows/build.ps1` 编译通过（0 错误、0 新增 `/W4` 警告）；`LichtFeld-Studio` Release（`137/137`）与 `LFS_TRAINER_ONLY` 均编译通过，核心相关 23 个 `lichtfeld_tests.exe` suite `313/313` 通过、Python 测试 `280/280`（3 skipped）通过。
  - `展厅` 实景全流程输出至 `work/展厅-ssi`（耗时 `617.10 s`，`588/588` 注册，MoGe-2 SSI 导出 `98.01 s`）与非破坏性 `export` 至 `work/展厅-ssi-verify` 均通过 `geosunpre check`（`passed: true`，`415,362,326` 有效深度与法向像素，`iterations=29000`、`max_cap=2600000`、`depth_loss_mode="ssi"`）。
  - `geosun_3dgs_train.exe`（5 步）与 `LichtFeld-Studio.exe`（1 步）实跑加载 `work/展厅-ssi-verify` 退出码 `0`：确认 `Depth anchors` 以 `ssi-depth` 拟合 `579/588` 相机（`mean |corr| = 0.985`）、`588` 相机启用卷帘渲染、`roi.json` 生效。
- **未验证项**：尚未跑完 `展厅-ssi` 或 `小场景` 的完整 29k–30k 步训练与留出评测对比。

## 2026-11-23：借鉴 Spirula 第一顺位轻量优化（`WriterPool` / 特征压缩 / 中间 BA 早停 / LAS 表面命中率保护）

- **核心改动**：
  1. **`masks` 固定写盘线程池与全图合成异步化**（[frontend.h](../src/frontend/frontend.h)、[frontend.cpp](../src/frontend/frontend.cpp)、[pipeline.cpp](../src/app/pipeline.cpp)）：以有界多线程池 `WriterPool`（1–8 线程，`workers` 控制）替换原 `AsyncWriter`，将全分辨率 `cv::resize` 上采样、镜头有效域合成与 PNG 写盘从主推理循环下沉至后台与 TensorRT 推理完全重叠。
  2. **SfM 建图前特征压缩**（[reconstruction.cpp](../src/sfm/reconstruction.cpp)）：在 `VerifyPairs` 之后、`Tracks` 构建之前新增 `FeatureCompaction`，剔除未出现在任何已验证匹配（及断点已绑点）中的冗余特征并重映射索引（统计写入 `cache/sfm.json` 的 `compact_features`），写盘 `incremental.bin` 断点与终局 `visual.bin` 前自动还原完整原始索引。
  3. **增量期中间 BA 松容差早停**（[optimization.h](../src/sfm/optimization.h)、[optimization.cpp](../src/sfm/optimization.cpp)、[reconstruction.cpp](../src/sfm/reconstruction.cpp)）：为 `Optimize` 增加可选 `function_tolerance`；仅在增量增长期临时 BA（每 8 帧局部 `1e-3`、导航转换/32 帧首次标定/周期及停滞全局 `5e-4`）放宽早停阈值，seed 与终局严格 BA（`1e-5` / `1e-6`）及 `shutter` 保持不变。
  4. **`align` LAS 表面命中率失位图硬保护**（[alignment.cpp](../src/alignment/alignment.cpp)）：借鉴 Spirula `5% + 20 cm` 规则新增 `CheckLasSurfaceHitRate`，在抽样未命中 `< 50%` 时自动对未命中点执行一次流式全量 LAS 复核防误删，统计记入 `cache/alignment.json` 的 `las_surface_check`。
- **已观测验证**（`.\tools\windows\build.ps1` Ninja Release `/W4` 0 错误 0 新增警告；`小场景` 296 图全流程输出至 `work/小场景-spirula-p1-out`，总耗时 `324.02 s`，`run.ps1 check` `passed: true`）：

| 指标 (`小场景` 296 图) | 基线 (`小场景-imu-sfm-out`) | 本次 (`小场景-spirula-p1-out`) |
| --- | :---: | :---: |
| `masks` 总耗时 / 主线程后处理 / 后台写盘 | — | `74.85 s` / `0.09 s` / `31.65 s`（全重叠） |
| `sfm` 特征压缩 `before -> after` | — | `909,312 -> 712,472`（`78.35%` 保留） |
| `sfm` 总耗时（中间 BA / 终局 BA） | `58.84 s` | `59.38 s`（44 次中间 `12.23 s` / 5 次终局 `30.49 s`） |
| 终局 `principal` BA 线性求解 | — | `33` 轮 `22.51 s`（线性 `13.92 s` / `8045` CG 迭代） |
| `align` `las_surface_check` | — | `296/296` 通过，`min_hit_rate = 91.25%`，`dropped = 0` |
| 最终注册 / `check` | `296 / 296` | `296 / 296`（`passed: true`） |
| 交付点数 / 观测数 | `122,614 / 413,944` | `121,670 / 419,989`（观测 `+1.46%`） |
| 交付长轨迹 (`tracks >= 3 / >= 8`) | `57,300 / 6,675` | **`57,954 / 6,903`**（`+1.1% / +3.4%`） |
| 交付重投影 `median / p90` | `0.7247 / 1.7625 px` | `0.7349 / 1.7777 px` |
| 导航位置残差 `median / p90` | `0.0421 / 0.0508 m` | `0.0436 / 0.0513 m` |
| `align` LAS 点到面 `median / p90` | `7.17 / 27.03 mm` | **`6.57 / 22.64 mm`** |
| 卷帘 `used` / `readout_s` | `true` / `[-0.04192, +0.03862]` | `true` / `[-0.04202, +0.03824]` |
| 卷帘留出 `before -> after` 中位 | `1.1650 -> 0.9534 px` | `1.1683 -> 0.9575 px` |
| `polish` 重关联 LAS 平面 `med / p90` | — | `4.04 / 13.84 mm` |

- **结论与未验证项**：采用为主线。`masks` 写盘与上采样已 100% 与 TRT 推理重叠，SfM 有效轨迹与长轨迹稳步增加、`align` 平面精度略优且卷帘/验收稳定通过；实测同时显示 `小场景` `sfm` 终局严格 BA 的线性求解（`13.92 s` / `8045` CG 迭代）为下一步 CUDA GPU BA 的主要提速空间。尚未在更大图量场景（如 `展厅` 588 图）实测增长期 BA 早停/特征压缩的耗时差异，亦未构造故意失位相机用例单独触发 `las_surface_check` 全量复核与剔除分支。

## 2026-11-24：修复参考图像离群阻断 SfM 进入导航坐标系及卷帘重锚负深度崩解（`南非展会`）

- **故障现象**：`work/南非展会.work`（1,073 幅异步双鱼眼，`L/R` 交替约 `65 ms`）在 `shutter` 阶段崩溃：`ScreenAlongAxis` 初始代价 `5.72e8`，双轴预筛与完整卷帘 BA（初始代价 `6.14e16`）均未通过，随后 `GlobalRigFallback` 抛出 `BA requires initial depth > 1e-6 (point 7, image 1039)`。
- **根因链路定位**：
  1. **主因（[reconstruction.cpp](../src/sfm/reconstruction.cpp) `to_navigation`）**：SfM 种子（`gauge_image = 73` `00078-L`、`scale_image = 77` `00080-L`）位于狭窄走廊起点（`L` 42–92 帧沿直线行进，导航横轴方差 `0.0629 < 0.09`），在进入主展厅时跨重复展位产生约 `3.55 m` 平移错位；主展厅注册图像超过九成后 `RobustSimilarity` 已识别出这 72 帧 `L` 错位簇与 `frame 197 R`（偏离 `14.39 m`），但 `to_navigation()` 因 `gauge_image`/`scale_image` 在离群集合中而永久拒绝转换，导致 `navigation_frame.used = false`，`1,073` 幅图像均未受 `NavigationGate` 与 `Position` 先验约束。`align` 中错位帧占 `6.8%` 未触发 `p90` 与 `las_surface_check`（`min_hit_rate = 0.7296`）拦截。
  2. **次因（[shutter.cpp](../src/sfm/shutter.cpp)）**：`ScreenAlongAxis` 未先运行 `ReanchorRigOutliers` 即施加无 Huber 二次 `RigLink`；`ReanchorRigOutliers` 将 79 个 rig 离群姿态拉回后产生 3,990 个负深度点，且直接对 `p.xyz` 调用无位移/正深度保护的 `RefinePoint`，将仅在错位簇内观测的点（如点 7）解算至极大距离导致 `PlaneCost` 代价爆炸（`6.14e16`）；`GlobalRigFallback` 也未筛选 `HasPositiveReferenceDepth` 即调用 `Optimize`。
- **改动**：
  - [reconstruction.cpp](../src/sfm/reconstruction.cpp)：`unregister` 同步清理 `recent`；`to_navigation` 在 `!outliers.empty()` 时校验剩余内点簇的最小数量（`>= 8`）、展开度与共享三维点支撑（`>= 24`），允许撤销含 `gauge_image`/`scale_image` 在内的离群图像并从内点中重选基线最宽的参考图像对及更新 `gauge_baseline`；外层补充停滞/全注册兜底（`allow_majority` 严格多数内点转换并重注册离群簇）。
  - [shutter.cpp](../src/sfm/shutter.cpp)：`ReanchorRigOutliers` 在临时向量上执行 `RefinePoint` 并校验有限性、位移上界（`max(5 m, 4 * max_shift)`）及全观测正深度（`z > 1e-3`）；`ScreenAlongAxis` 前置 `ReanchorRigOutliers` 并过滤 `HasPositiveShutterDepth`；`GlobalRigFallback` 仅向 `Optimize` 传入 `HasPositiveReferenceDepth` 轨迹并在优化后清理残余负深度轨迹。
- **实景验证**（`.\tools\windows\build.ps1` Ninja Release `/W4` 0 错误 0 新增警告；复用前端缓存于 `work/南非展会-navfix.work` 从 `sfm` 从头跑至 `export` 并保留于 `work/南非展会-navfix`）：

| 指标 (`南非展会` 1,073 图) | 修复前 (`work/南非展会.work`) | 修复后 (`work/南非展会-navfix.work` / `南非展会-navfix`) |
| --- | :---: | :---: |
| `sfm.navigation_frame.used` / 触发点 | `false` | `true`（`700` 图处撤销 `70` 离群，重选参考 `[445, 6]`） |
| 最终注册 / `check` | `1073/1073`（含 `73` 帧 `> 0.25 m` 错位）→ `shutter` 崩溃 | **`1073/1073`**，`geosunpre check` **`passed: true`** |
| `aligned.bin` `\|center - body_pos\|` (med / p90 / max) | max `14.3859 m`（`73` 帧 `> 0.25 m`） | `0.0745 / 0.1005 / 0.1197 m`（**`0` 帧 `> 0.25 m`**） |
| `align` `las_surface_check` 最低命中 / `joint.used` | `0.7296` / `true` | `0.8787` / `true`（位移 `31.9 / 64.8 mm`） |
| `shutter.used` / `readout_axis` / `readout_s` | 失败退出（`6.14e16` → depth 异常） | **`true`** / `rows` / `[-0.03888, +0.03822] s` |
| 卷帘留出重投影 `before -> after` (med / p90) | — | `1.2842 -> 1.0683 px` / `3.1794 -> 2.8027 px` |
| 卷帘 `explained_rotation` (L / R) / `polish` | — | `0.8481 / 0.9096`（`imu`）/ `true`（`6.69 / 18.04 mm`） |
| 交付点数 / 观测 / 重投影 `med / p90` | — | `321,218` / `1,698,452` / `0.8960 / 1.9976 px` |

- **结论**：修复确立为主线；未在 `GlobalRigFallback` 真实命中场景复测该分支。

## 2026-11-25：Ceres CUDA BA、TensorRT 动态批推理、精确 `SO(3)` 卷帘与 IMU 时差自标定、导航引导种子选对

- **改动概要**：
  1. **Ceres CUDA 混合精度 `DENSE_SCHUR` BA**（[dependencies.cmake](../dependencies/dependencies.cmake)、[optimization.cpp](../src/sfm/optimization.cpp)、[build.ps1](../tools/windows/build.ps1)）：开启 Ceres 2.2.0 `USE_CUDA=ON`（链接 `cudart`、`cublas`、`cusolver`、`cusparse` 并随 `build.ps1` 复制 6 个运行时 CUDA DLL）；对 `>= 64` 图且含可变三维点的全局 BA，在 Schur 补列数 `<= 12000` 且空闲显存满足 `>= 2 * (12 * schur_cols^2) + 512 MiB` 时启用 `DENSE_SCHUR + CUDA + use_mixed_precision_solves=true`（FP32 Cholesky + FP64 迭代精化），否则回退 `ITERATIVE_SCHUR + SCHUR_JACOBI`。
  2. **TensorRT 动态批推理**（[networks.py](../tools/models/networks.py)、[prepare.py](../tools/models/prepare.py)、[inference.cpp](../src/frontend/inference.cpp)、[frontend.cpp](../src/frontend/frontend.cpp)、[monocular.cpp](../src/geometry/monocular.cpp)、[supervision_kernels.cu](../src/geometry/supervision_kernels.cu)）：为 LoMa `matcher`（`B=1..8`, `opt=8`）与 MoGe-2 `geometry`（`B=1..8`, `opt=7`）开启动态批维度，构建独立模型包 `models-batched`（保留只读 `models/` 不变）；运行时按引擎 `MaxBatch` 与可用显存（`< 5 GiB` 收敛至 `<= 4`）自动分批。
  3. **精确 `SO(3)` 卷帘模型与暴露–IMU 时钟偏置自标定**（[model.h](../src/geometry/model.h)、[optimization.cpp](../src/sfm/optimization.cpp)、[shutter.cpp](../src/sfm/shutter.cpp)、[scene.cpp](../src/io/scene.cpp)、[monocular.cpp](../src/geometry/monocular.cpp)、[supervision_kernels.cu](../src/geometry/supervision_kernels.cu) 及下游 `LichtFeld-Studio` [rolling_shutter.cpp](file:///d:/Dev/LichtFeld-Studio/src/training/components/rolling_shutter.cpp)）：将线性一阶卷帘统一升级为精确 `SO(3)` `Exp(s[ω]×)(v - s·τ·u)` 与精确左雅可比 `-s·J·[v]×·J_l(sω)`（经中心有限差分校验 `MAX_ABS_ERR = 1.84e-7`，`MAX_REL_ERR = 1.11e-7 < 1e-6`），单轴旋转边界由 `0.05` 放宽至 `0.10` rad；并在 `shutter` 前对 `[-80, +80] ms` 自标定共享时钟偏置 `time_offset_s`。
  4. **导航展开度与 IMU 旋转感知的三级种子选对**（[reconstruction.cpp](../src/sfm/reconstruction.cpp)）：按 IMU 旋转一致性（`<= 3.0°`）、强邻域（`inliers >= 48`）数量及位置协方差横向展开度 `transverse_m` 分 Tier 2 / 1 / 0 择优。

### Ceres BA 线性求解器对比（同题复现，`ba_workers = 8`）

Ceres 2.2.0 在 `LAPACK=OFF`、`SUITESPARSE=OFF` 下，所有纯 CPU 替代求解器（`SPARSE_SCHUR`、`DENSE_SCHUR`、`CLUSTER_TRIDIAGONAL`、`CLUSTER_JACOBI`）均慢于 `ITERATIVE_SCHUR + SCHUR_JACOBI`（无 LAPACK 时 Eigen 稠密求解与无 AMD/METIS 重排的 `EIGEN_SPARSE` 在全局内参耦合块上出现严重填充/慢分解）；开启 `USE_CUDA=ON` 后 `DENSE_SCHUR (CUDA Mixed)` 在大规模场景获得显著线性求解加速且终态代价与重投影误差完全一致：

| 求解器配置 | `小场景` A (`principal`, 296 图) 总耗时 / 线性 | `小场景` B (`align`) 线性 | `小场景` C (`shutter+polish`) 线性 | `南非展会` A (`principal`, 1,073 图) 总耗时 / 线性 | `南非展会` C (`shutter+polish`) 总耗时 / 线性 |
| --- | :---: | :---: | :---: | :---: | :---: |
| `ITERATIVE_SCHUR + SCHUR_JACOBI`（基线） | `8.87 s` / `4.61 s` | `3.27 s` | `5.97 + 5.49 = 11.46 s` | `39.23 s` / `24.73 s` | `55.07+62.53=117.60 s` / `35.73+40.96=76.69 s` |
| `DENSE_SCHUR`（CPU Eigen） | `10.42 s` / `5.80 s` | `4.18 s` | `11.71 + 8.66 = 20.37 s` | — | — |
| `SPARSE_SCHUR`（CPU `EIGEN_SPARSE`） | `11.55 s` / `6.76 s` | `5.78 s` | `17.89 + 12.91 = 30.80 s` | `> 220 s`（第 9 轮中止） | — |
| `ITERATIVE_SCHUR + CLUSTER_TRIDIAGONAL` | `20.21 s` / `15.68 s` | `9.32 s` | `50.67 s`（仅 `shutter`，后中止） | — | — |
| `ITERATIVE_SCHUR + CLUSTER_JACOBI` | `29.12 s` / `24.50 s` | — | — | — | — |
| `DENSE_SCHUR`（CUDA FP64） | **`8.04 s` / `3.72 s`** | `3.51 s` | `6.28 + 4.79 = 11.07 s` | `32.64 s` / `18.33 s` | `48.42+50.27=98.69 s` / `28.19+29.66=57.85 s` |
| **`DENSE_SCHUR`（CUDA Mixed，采用）** | `9.02 s` / `4.31 s` | **`2.43 s`** | **`5.83 + 4.60 = 10.43 s`** | **`31.60 s` / `16.87 s`**（`-31.8%`） | **`44.41+45.35=89.76 s` / `23.39+23.77=47.16 s`**（`-38.5%`） |

### 三场景全流程实景对比（`models-batched`，交付保留于 `work/*-opt24`）

三套实景均通过 `geosunpre check`（`passed: true`，注册率 100%）：

| 指标 | `小场景` 基线 | `小场景-opt24` | `展厅` 基线 | `展厅-opt24` | `南非展会` 基线 | `南非展会-opt24` |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| `matches` 耗时 (`B=1..8`) | `25.98 s` | **`20.41 s`**（`-21.4%`） | `73.60 s` | **`51.01 s`**（`-30.7%`） | `99.78 s` | **`78.38 s`**（`-21.4%`） |
| 种子选对 / 导航入系点数 | — / `11`（`0` 撤销） | Tier 2 `spread=0.94 m` / `11`（`0` 撤销） | — / `9`（`0` 撤销） | Tier 2 `spread=1.00 m` / `10`（`0` 撤销） | — / `700`（`70` 走廊离群） | **Tier 2 `spread=2.04 m` / `133`（`0` 离群）** |
| `sfm` 终局 BA 总耗时 / 线性 | `29.08 s` / `13.43 s` | `33.09 s` / `13.90 s` | `86.71 s` / `50.02 s` | **`76.92 s` / `40.27 s`** | `156.43 s` / `86.21 s` | **`140.39 s` / `63.35 s`**（`-26.5%`） |
| `align joint` BA / 线性 | `12.53 s` / `8.10 s` | **`5.92 s` / `2.45 s`**（`-69.8%`） | `19.68 s` / `14.76 s` | **`17.87 s` / `12.30 s`** | `117.12 s` / `90.05 s` | **`29.28 s` / `14.95 s`**（**`6.0×`**） |
| IMU 时差 `time_offset_s`（视轴中位） | — | `0.0 ms`（`used=false`，`0.160°`） | — | **`-13.0 ms`**（`0.371° → 0.268°`） | — | **`-5.5 ms`**（`0.232° → 0.209°`） |
| `shutter` + `polish` BA 线性 | `6.57 + 7.64 = 14.21 s` | **`5.40 + 4.98 = 10.38 s`** | `27.61 + 41.32 = 68.93 s` | **`22.85 + 21.68 = 44.53 s`**（`-35.4%`） | `46.64 + 47.91 = 94.55 s` | **`24.50 + 24.66 = 49.16 s`**（`-48.0%`） |
| 留出重投影中位 (`before → after px`) | `1.1516 → 0.9376` | `1.1660 → 0.9421` | `1.4188 → 1.1337` | `1.4340 → 1.1528` | `1.2842 → 1.0683` | **`1.2523 → 1.0547`** |
| LAS 平面误差 `plane_after` / 重关联 (mm) | `4.25/15.38` / `4.18/14.75` | **`3.97/14.79` / `3.92/14.34`** | `5.94/17.49` / `5.90/16.81` | **`5.58/16.98` / `5.53/16.38`** | `6.83/18.41` / `6.69/18.04` | **`6.32/18.51` / `6.18/17.91`** |
| MoGe-2 `moge_estimate`（`host_s`） | `47.35 s` (`33.29 s`) | `46.68 s` (**`2.34 s`**) | `96.97 s` (`64.51 s`) | `92.41 s` (**`3.35 s`**) | `174.86 s` (`120.09 s`) | `169.17 s` (**`5.16 s`**) |
| 交付点数 / 重投影 (`med / p90 px`) | `123,021` / `0.7226 / 1.7592` | `122,597` / `0.7327 / 1.7703` | `154,185` / `0.8908 / 2.0720` | `154,243` / `0.8929 / 2.0667` | `321,218` / `0.8960 / 1.9976` | **`334,070` / `0.8741 / 1.9639`** |

- **下游验证**：`LichtFeld-Studio` `RollingShutter*` 单元测试 `8/8` 通过；`LichtFeld-Studio.exe` 加载 `work/展厅-opt24` 运行 5 步无头冒烟训练退出码 `0`（确认 `588` 相机启用行读出卷帘、加载 `588/588` 深度与法向先验及 `roi.json` 52 顶点裁切盒，训练完成）。
- **结论与未验证项**：采用全部优化为主线；注意 MoGe-2 ViT-L 批推理将主机调度等待由 `33–120 s` 压缩至 `2.3–5.2 s`，但因 RTX 4090 上单脸 ViT-L 本身已达算力饱和，端到端 `moge_estimate` 仅微降 `1%–5%`。尚未运行 `opt24` 交付包的完整 20k–41k 步 3DGS 训练对比指标，亦未在 `< 8 GB` 显存卡或超过 `1,330` 幅卷帘图的显存回退分支上实测。
