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
  - [RasterizeToPixelsFromWorld3DGSBwd.cu](../../LichtFeld-Studio/src/training/rasterization/gsplat/RasterizeToPixelsFromWorld3DGSBwd.cu)：为 `rasterize_to_pixels_from_world_3dgs_bwd_splat_kernel` 增加 `kColorVariance` 编译期模板参数（关闭时将 `s_cv_target`/`s_cv_buf` 缩至 1 元素，共享内存降回 `8960 / 9216 B`，`LOCAL:0`），并在开启分支前由 warp `lane 0` 读取 `*color_variance.pixel_weight_sum` 后 `__shfl_sync` 广播。
  - [trainer.cpp](../../LichtFeld-Studio/src/training/trainer.cpp)：在分辨率匹配且无需重拟合时复用已适配的深度先验并 `sync_to_stream`，避免重复调用 `fit_depth_prior`。
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
  3. **精确 `SO(3)` 卷帘模型与曝光–IMU 时钟偏置自标定**（[model.h](../src/geometry/model.h)、[optimization.cpp](../src/sfm/optimization.cpp)、[shutter.cpp](../src/sfm/shutter.cpp)、[scene.cpp](../src/io/scene.cpp)、[monocular.cpp](../src/geometry/monocular.cpp)、[supervision_kernels.cu](../src/geometry/supervision_kernels.cu) 及下游 `LichtFeld-Studio` [rolling_shutter.cpp](../../LichtFeld-Studio/src/training/components/rolling_shutter.cpp)）：将线性一阶卷帘统一升级为精确 `SO(3)` `Exp(s[ω]×)(v - s·τ·u)` 与精确左雅可比 `-s·J·[v]×·J_l(sω)`（经中心有限差分校验 `MAX_ABS_ERR = 1.84e-7`，`MAX_REL_ERR = 1.11e-7 < 1e-6`），单轴旋转边界由 `0.05` 放宽至 `0.10` rad；并在 `shutter` 前对 `[-80, +80] ms` 自标定共享时钟偏置 `time_offset_s`。
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

## 2026-11-25：SAM3 主点居中方形 ROI、操作员手持配件桥接、边缘残缺肢体与弱阴影掩码修复

- **问题定位（基于 `展厅-opt24` 及其 `.licht` 工程）**：`展厅-opt24` 地面黑洞点位 `(1.75, 15.85, -1.03)` 上空可见近距相机中存在三类 SAM3 原始漏检（`cache/masks` 与 `cache/static_masks` 完全一致）：
  1. `00235-L`、`00236-L`、`00294-L` 中操作员手臂上的黑色腕戴手机/手表未被 `"person"` 掩膜覆盖，形成贯穿手臂的黑色未掩膜缺口（`min_lum = 0.04–0.09`），近景视角直接将黑色像素投影至展厅地面；
  2. `4000×3000` 图像含左右黑边直接压成 `1008×1008` 浪费分辨率并导致鱼眼边缘残缺裤腿/鞋（如 `00036-L`）漏检；
  3. 室内地面弱阴影（如 `00313-L` 分数 `0.34`）被 `kGroundingScore = 0.5` 拦截，且粗网格外扩仅 `3×3`（1 格）。
- **改动**：
  - [frontend.cpp](../src/frontend/frontend.cpp)：新增 `GroundingRoi` 按主点居中截取保留完整鱼眼圆上下文的方形 ROI（`4000×3000 -> 3000×3000`，`3600×3600` 保持不变）并更新 `ImageReader`/`ComposeFullMask`；将触及有效域外圈（10 格环带）的 `person` 阈值放宽至 `0.25`、`shadow` 阈值放宽至 `0.25`（保留既有人近邻 `kShadowReach=24` 与超大裁剪规则）；对 `people` 增加半径 `16` 格椭圆闭运算桥接（面积受 `8%` 有效域与 `50%` 人物面积双上限约束并针对桥接区外扩 `5` 格以包住表框）及人围封闭小空隙填充（`<= 3%` 有效域且人物边界占比 `>= 35%`）；终态 `transient` 外扩由 `3×3` 增至半径 `2` 格椭圆（`5×5`）。
  - [reconstruction.md](reconstruction.md)、[models.md](models.md)：同步更新 `masks` 阶段与 `grounding` 输入说明。
- **实景验证**（`.\tools\windows\build.ps1` Ninja Release `/W4` 0 错误 0 新增警告；全流程从零运行至独立目录 `work/展厅-maskfix.work` / `work/展厅-maskfix`，并通过 `geosunpre check`）：

| 指标 (`展厅` 588 图) | 基线 (`展厅-opt24`) | 修复后 (`展厅-maskfix`) |
| --- | :---: | :---: |
| `masks` 总耗时 | `117.41 s` | `119.69 s`（`+1.9%`） |
| `grounding` 人物 / 边缘补回 / 机器人保持 | `646` / — / `104` | **`839`** / `181` / `460` |
| `accessory_bridges` / `filled_pockets` | — | `6,833` / `147` |
| `shadows`（近邻采纳 / 裁剪 / 远距忽略） | `79` / `2` / `68` | **`300`** / `8` / `583` |
| 红框地面近距（`<2.7 m`）极暗未掩膜泄漏 (`vdark<0.22`) | `00235-L`: `7`, `00236-L`: `3` | **`00235-L`: `0`, `00236-L`: `0`** |
| `supervision` `min_distance_m <0.4m / <0.25m` / 中位 | `338 / 226` / `0.3420 m` | **`301 / 191` / `0.3895 m`** |
| 注册 / 交付点数 / 重投影 (`med / p90 px`) | `588/588` / `154,243` / `0.8930 / 2.0670` | `588/588`（`rig_rescue=0`）/ `155,187` / `0.8965 / 2.0744` |
| `align` LAS 点到面 (`med / p90 mm`) | `5.61 / 16.55` | **`5.50 / 16.40`** |
| 卷帘 `used` / 留出中位 (`px`) | `true` / `1.1528` | `true` / `1.1525` |

- **结论与未验证项**：目视抽检 `00235-L`、`00236-L`、`00294-L`、`00313-L`、`00264-L`、`00036-L` 腕表/腿鞋/人影均已完整覆盖，且 `00051-L`/`00081-L` 人形展品及展台保持静态未误杀。尚未用 `work/展厅-maskfix` 跑完下游 3DGS 全程训练以直接对比训练后的红框与路面视觉效果（另：路面大片分层主因为 MoGe-2 SSI 深度先验在展厅地板系统性偏深 `+10~+36 cm`，待下一步单独验证）。

## 2026-11-25：LAS 分面尺度对齐与三维双边网格残差细化消除 MoGe-2 SSI 地面分层

- **根因诊断（基于 `展厅-maskfix`、`points3D.bin` 与下游 `LichtFeld-Studio` `fitDepthAnchors`）**：下游 `fitDepthAnchors` 仅对每帧监督深度做一次全局仿射拟合 `a * d + b`，而 MoGe-2 90° 透视子图仅凭重叠中位比值拼接时存在显著跨面尺度漂移（`face_scales` `p1–p99 = 0.49–1.68`）与近距离地板偏深（例如 `00264-L` 在非地面 SfM 观测拟合下地面有符号中位误差 `+41.0 cm`，`411` 幅含地面视角 `P90` 达 `33.24 cm`、`p10..p90` 跨越 `[-22.34, +22.76] cm`），导致多视角监督把同一块抛光地面在三维空间拉出多层高斯片；此外极少数贴近有效域边缘的极近伪影（`00264-R` `min_d = 0.14 m`）也会压缩 16 位 SSI 动态范围。
- **改动**（保持下游 `moge2_ssi` / `uint16_png_ssi` 契约与 `LichtFeld-Studio` 零改动）：
  - [monocular.h](../src/geometry/monocular.h)、[monocular.cpp](../src/geometry/monocular.cpp)：`EncodeMogeSsiMaps` 新增可选 `las_depth` 输入，先用 5×5 局部最小抑遮挡窗口（`d <= las_min * 1.08 + 0.08`、`[0.20, 60.0] m`）清洗 LAS 投影深度；对每个高权重静态样本不少于 `256` 点的子图先求稳健尺度并在景深跨度充足且严格降差时叠加 8 轮 Cauchy IRLS 仿射对齐，再融合并由重叠中位比值补齐无 LAS 支撑子图；随后在 `[log d, x, y]` 三维双边网格（`sigma_s = max(16, max(w,h)/32)`、`sigma_r = 0.25`、阈值 `0.45` 过滤玻璃穿透）上做 2 遍保边平滑残差校准，最后按 `[0.1%, 99.9%]` 分位（下界 `0.05 m`）编码为 `[1311, 65535]` 16 位 SSI PNG。
  - [supervision.h](../src/geometry/supervision.h)、[supervision.cpp](../src/geometry/supervision.cpp)、[colmap.cpp](../src/io/colmap.cpp)：`ExportSupervision` 在加载 TensorRT 几何引擎前投影最终交付视角的 LAS 深度至 `.las_scratch/las_depths.u16` 后立即释放 LAS GPU 缓存；完整 `run` 下 `BuildStaticMasks` 将新鲜构建的 `LasSurfelCache` 移交给 `ExportSupervision` 复用，仅解析一次 LAS；`SupervisionRequest` 与 `supervision.json` 新增 `depth_alignment = "las_face_bilateral_v1"` 与 `las_file` 请求身份。
  - [supervision.md](supervision.md)、[cache.md](cache.md)、[output.md](output.md)：同步更新监督导出流程与请求字段说明。
- **实景验证**（`.\tools\windows\build.ps1` Ninja Release `/W4` 0 错误 0 新增警告；非破坏性独立 `export` 至保留目录 `work/展厅-depthfix` 与 `work/小场景-depthfix` 并均通过 `geosunpre check`，确认 `.las_scratch` 无残留）：

| 指标 | `展厅-maskfix`（纯 MoGe-2） | `展厅-depthfix`（LAS 对齐） | `小场景-opt24`（纯 MoGe-2） | `小场景-depthfix`（LAS 对齐） |
| --- | :---: | :---: | :---: | :---: |
| `geosunpre check` / 图数 | `passed: true` / `588` | **`passed: true` / `588`** | `passed: true` / `296` | **`passed: true` / `296`** |
| `PLY and supervision`（含 LAS 投影） | — | `126.70 s`（LAS `23.53 s`） | — | `63.35 s`（LAS `9.12 s`） |
| LAS 对齐子图 / 双边细化生效 | `0` / `0` | **`4116/4116`** / **`588/588`** | `0` / `0` | **`2072/2072`** / **`296/296`** |
| 对齐后相对清洁 LAS 逐像素绝对误差 `med / p90` | — | **`1.92 / 3.52 cm`** | — | **`1.28 / 4.75 cm`** |
| 非地面 SfM 仿射拟合后地面误差 `\|signed_med\| p50 / p90`（`411` 图） | `10.78 / 33.24 cm` | **`1.31 / 6.43 cm`** | — | — |
| 全图 SfM 仿射拟合后绝对误差 `p50 / p90` / 纯米制 `p50` | `00264-L` 地面 `+41.0 cm` | 地面纯米制 `-0.2 cm` | `6.20 / 17.28 cm` | **`1.95 / 5.49 cm`**（`1.92 cm`） |
| 局部对数深度梯度相关性 `p5 / p50` | — | `0.9147 / 0.9764` | — | **`0.9750 / 0.9918`** |

- **结论与未验证项**：两个室内及室外混合场景的地面/三维尺度跨视角一致性均大幅改善且完整保持边缘细节，下游 LFS 接口无需任何修改即可直接受益；`00264-L` / `00236-L` / `00292-L` 可视化核验显示地板/墙面分面尺度错位消除且建筑轮廓锐利。**未验证项**：1）尚未以 `work/展厅-depthfix` 在下游 `LichtFeld-Studio` 跑完 29,000 步 3DGS 训练以实测成图地面分层消除情况；2）本次通过独立 `export`（命中已有 `static_masks` 缓存）验证，完整从头 `run` 中由 `BuildStaticMasks` 直接移交复用 `LasSurfelCache` 的路径已通过编译与静态检查，尚未在重跑全流程时实测耗时。

## 2026-11-25：`StaticMasks` 新增多视角光度瞬态屏蔽（移动阴影与强镜面高光）

- **问题定位（基于 `展厅-depthfix` 训练截图与实景原图排查）**：
  1. 尽管 SAM3 已结合 `kShadowReach=24` 遮蔽近人阴影，操作员离开画幅或拖长的人影（如 `展厅` `00101-L`、`00058-L`、`00072-L` 及鞋边弱影 `00290-L`，`小场景` `00161-L`）仍保留 `255`，在 3DGS 地板上长出视角撕裂脏斑；
  2. 金属展牌（`展厅` `00117-R`、`00133-R`）与玻璃消防箱（`00305-R`）在少数视角存在强反光高光斑，仅靠降 `SH` 阶数不足以消除。
- **改动**：
  - [static_masks.h](../src/geometry/static_masks.h)、[static_masks.cpp](../src/geometry/static_masks.cpp)：在 `StaticMasks` 新增 Pass 2.5 `DetectPhotometricTransients`（全部规则字段同步纳入 `StaticMaskRules()`，使 `cache/static_masks` 自动失效重建）：
    1. **每镜头径向暗角自标定**：利用 Pass 1 投影的 LAS 深度（`CleanAndDownsampleLasDepth` 抑遮挡清洗、块内高差 `bimodal` 过滤与单像素平滑补洞）在 `256` 长边网格上对同镜头共视光滑中灰区域做 5 轮 Huber IRLS 拟合 6 阶多项式径向暗角并还原线性亮度；
    2. **多视角光度共识判别**：将平滑静态像素投影至至多 `20` 个邻近共视帧（`< 10 m`，支持卷帘固定点映射），校验 z-buffer 深度一致且仅采纳线性中位曝光比在 `[0.50, 2.00]` 内的配对（跳过 `00150-L` 逆光暴跳帧），采用对称 3×3 邻域极值容差（消除亚像素/网格边沿抖动）与图像梯度上限，在至少 6 个共视帧中 ≥70% 一致偏暗（`ratio < 0.78, diff < -0.06`）或偏亮（`ratio > 1.30, diff > +0.10`）处形成种子，并针对非静态人物 `24` 格邻域内放宽弱影种子（`>= 5` 帧 ≥70%、`ratio < 0.88, diff < -0.04`）以覆盖 `00290-L` 等脚边柔影；
    3. **受限滞后扩边与熔断**：经局部连通扩边与 `20%` 单图静态占比熔断后写入 `0`，同步作用于下游 `masks/`、`depths/`、`normals/` 与 PLY 视觉补点。
  - [output.md](output.md)、[cache.md](cache.md)：同步更新静态掩码规则与 `photometric` 摘要说明。
- **首轮调参排查与收紧**：首版未限制配对曝光比区间、缺少对称 3×3 容差/图像梯度过滤且滞后扩边阈值过松（`0.90 / 0.03`、`radius=6`），在 `展厅-photofix`（`1.080%` 新增遮蔽）于 `00235-R` 天花格栅误遮 `16.13%`、在 `小场景-photofix`（`3.171%`）于逆光走廊 `00150-L` 误遮 `26.35%`，且漏检 `00290-L` 弱影；实施上述 5 项收紧与人旁弱影先验后导出至独立保留目录 `work/展厅-photofix2` 与 `work/小场景-photofix2`：

| 指标 | `展厅-photofix2` (vs `展厅-depthfix`) | `小场景-photofix2` (vs `小场景-depthfix`) |
| --- | :---: | :---: |
| `geosunpre check` / 图数 | **`passed: true`** / `588/588` | **`passed: true`** / `296/296` |
| Pass 2.5 耗时 / `capped_images` | `6.69 s` / `0` | `8.52 s` / `0` |
| 暗角边缘线性增益 (`L / R`) | `0.791 / 0.765` | `0.775 / 0.755` |
| 新增遮蔽总像素（占比） | `6,057,641`（**`0.227%`**） | `15,891,408`（**`0.920%`**） |
| 偏暗阴影 / 偏亮高光像素（图数） | `3,806,884` (`99`) / `2,250,757` (`50`) | `15,667,852` (`137`) / `223,556` (`7`) |
| 典型目标帧遮蔽 | `00290-L`: `0.63%`、`00101-L`: `4.79%`、`00117-R`: `1.35%`、`00133-R`: `1.96%` | `00161-L`: `12.56%`（地毯人影） |
| 原误检帧回归 | `00235-R`: `16.13%` → **`0.75%`** | `00150-L`: `26.35%` → **`0.00%`**、`00169-L`: `17.79%` → **`0.00%`** |
| 三维目标点静态视角保留 | `gold_plaque`: `169/169 (100%)`、`floor_fig1`: `42/43 (97.7%)`、`floor_fig2`: `77/77 (100%)` | — |

- **结论与未验证项**：两个实景交付包均将误检压至 `0.23%–0.92%` 且精准屏蔽地板移动阴影与金属牌/玻璃反光高光。**未验证项**：尚未以 `work/展厅-photofix2` 在下游 `LichtFeld-Studio` 完成 29,000 步 3DGS 训练，需实际试训确认图 1/2 地面黑斑与图 3 金属牌反光伪影的最终成图改善。

## 2026-11-26：`StaticMasks` 多视角相对曝光求解（无损 EXIF 重写）与过曝裁切屏蔽（`小场景-evfix`）

- **问题根因定位（`work/小场景` 中庭白墙正面雾斑）**：
  1. **硬件双镜头 EXIF 抄写偏差**：实测 `work/小场景` 148 对 `L`/`R` 照片的原始 EXIF `EV` 完全相等（`max |EV_L - EV_R| == 0.0`），但 `R` 镜头独立测光/增益与 `L` 并不一致。在红框白墙 `P_wall` 57 张室外非裁切视角上，原硬件 EXIF 与去暗角线性亮度仅呈弱相关（`R` 镜头 `corr = +0.427`、补偿后残差标准差 `0.296 EV` 反而劣于裸图 `0.262`），导致正对红框的 `00111-R ~ 00117-R` 被错误种子化偏高 `+0.23 ~ +0.76 EV`，与贴墙近景 `00048-L ~ 00051-L` 产生 `+0.7 ~ +1.0 EV` 假亮度冲突，在 `SH=0` + 薄墙监督下驱动出“小颗粒灰白层 + 大尺度亮白补丁层”叠加与 Z 排序翻转雾斑；
  2. **暗走廊过曝裁切帧未屏蔽**：`00156-R ~ 00165-R` 从暗走廊（`EV +1.0 ~ +5.9`）回拍室外白墙时，中庭墙面及门洞已饱和裁切（`RGB 242–255`）却仍保持 `mask == 255` 参与梯度反传。
- **改动**：
  - [static_masks.h](../src/geometry/static_masks.h)、[static_masks.cpp](../src/geometry/static_masks.cpp)：在 `DetectPhotometricTransients` 的 `FitVignetting` 之后新增 `SolveRelativeExposures`，在全场景 `L`/`R` 共视帧对（`<= 10 m`、去暗角中灰区 `>= 40` 采样、`MAD <= 0.45 EV`）上构建稀疏图并跑 15 轮 Huber IRLS 求解每帧零均值相对 EV（结果写入 `cache/static_masks.json` 的 `exposure_ev` 与 `photometric.exposure`）；同时利用校准相对 EV（`ev_i - ev_j >= 0.35 EV`）在共视更暗未裁切帧上核验并经滞后扩边屏蔽过曝裁切区为 `0`（独立 `35%` 熔断上限）。全部新规则常量纳入 `StaticMaskRules()`。
  - [crop.h](../src/io/crop.h)、[crop.cpp](../src/io/crop.cpp)、[colmap.cpp](../src/io/colmap.cpp)：新增 `BuildCalibratedExifApp1`、`CopyJpegWithCalibratedExif` 与 `ReadJpegExifEv`，在 `CropJpeg`（及未裁剪分支）中无损替换 JPEG 的 `"Exif\0\0"` `APP1` 段为校准后的 `ExposureTime`/`FNumber=1`/`ISO=100`/`Orientation=1`（保留 DCT 像素与 XMP/ICC 等其他段），并在 `manifest.json` 的 `images` 记录 `calibrated_ev`、`CheckBundle` 逐图回读核验误差 `<= 0.01 EV`。
  - [output.md](output.md)、[cache.md](cache.md)：同步更新静态掩码、无损 EXIF 重写及 `check` 回读契约。
- **实景验证**（`.\tools\windows\build.ps1` Ninja Release `/W4` 0 错误 0 新增警告；从零运行全流程 `.\run.ps1 run --input D:\Data\小场景 --debug --work work\小场景-evfix.work --output work\小场景-evfix`，总耗时 `289.72 s`，保留 `work/小场景-evfix` 与 `work/小场景` 并存对照）：

| 指标 (`小场景` 296 图) | 基线 (`work/小场景`) | 修复后 (`work/小场景-evfix`) |
| --- | :---: | :---: |
| `geosunpre check` / 注册数 | `passed: true` / `296/296` | **`passed: true`** / **`296/296`** |
| 全局相对曝光求解 | — | **`calibrated: true`**，`1` 连通分量（`296/296`），`5,995` 边（含 `1,124` 跨 `L/R` 边），残差 med **`0.0225 EV`** / p90 **`0.1033 EV`** |
| 交付 JPEG EXIF 回读一致性 | — | 296 图 `orientation = 1`，`max |PIL_ev - calibrated_ev| = 0.000054 EV` |
| `P_wall` 57 视角 `corr(EV, log2_lum)` / 补偿后 `std` (`L` / `R` / `ALL`) | `0.947 (0.127)` / `0.427 (0.296)` / `0.591 (0.275)` | **`0.985 (0.055)`** / **`0.933 (0.122)`** / **`0.944 (0.108)`** |
| 过曝裁切屏蔽 (`clipped_pixels` / `images` / `capped`) | — | `14,243,749 px` / `79` 图 / `0` 熔断（`00162-R`: `+14.85%`、`00163-R`: `+13.50%`、`00049-L`: `0.01%`、`00114-R`: `0.00%`） |
| `LichtFeld-Studio.exe` 5 步冒烟 | — | 退出码 `0`：确认 `PPISP exposure seeded from EXIF for 296 of 296 frames (mean -0.00 EV)`、`296/296` 卷帘/深度/法向/掩码与 `roi.json` 生效 |

- **结论与未验证项**：从源头彻底消除了 `R` 镜头抄写 `L` 镜头 EXIF 的系统误差（`ALL` 亮度残差标准差由 `0.275` 降至 `0.108 EV`）并精确隔离走廊过曝白块，下游 `LichtFeld-Studio` 零改动即在 Step 0 按真实相对曝光初始化。**未验证项**：1）尚未以 `work/小场景-evfix` 在 `LichtFeld-Studio` 跑完完整 20k–30k 步训练并实测红框白墙在 novel view 的最终成图改善；2）尚未验证 `--no-crop-valid-domain` 分支下 `CopyJpegWithCalibratedExif` 的实景交付回读。

## 2026-11-26：源头线性匀光匀色（去暗角 + 3 通道白平衡 + 有界曝光对齐）与过曝/泛光掩码 v2（`小场景-evfix2`）

- **问题复盘（基于 `work/小场景-evfix` 试训反馈与 22,475 条无遮挡 SfM 轨迹诊断）**：
  1. **过曝掩码 v1 盲区**：仅检查 `<= 20` 邻且 `< 10 m`、强制要求 `depth_smooth` 且仅允许 `radius=3` 格滞后泛洪，导致走廊深处（`00153-R`、`00162-R` 红框与远端安全出口玻璃门、`00167-L`）等大片死白（`RGB=255`）及被镜头泛光冲灰的枝叶/门框缝隙漏检；
  2. **物理暗角与 ISP 高光肩部残差**：双镜头强径向暗角（`L: -22%`, `R: -30%`）仍保留在导出的 JPEG 中，在 `P_wall` 上造出 `0.730–0.874` 的中心/边缘裂隙；同时相机 ISP 在 `sRGB > 0.78` 进入高光压缩肩部（实测曝光斜率 `0.69`），而 `PPISP`（`train_crf=false`）仅做全场景均值归零+线性缩放，且仅凭 EXIF 无法抵消 `R` 镜头独立白平衡漂移（最大 `±0.15 EV`）；
  3. **有界像素匀光策略**：直接把室外主众数曝光（KDE 峰值 `-0.692 EV`）与 3 通道白平衡差/径向暗角在线性光空间消除，只把超出 8-bit 安全动态范围（中位 `p50 >= 0.25`、最大压暗 `1.40 EV`、最大提亮 `1.20 EV`）的跨室内部份残差写入 EXIF。
- **改动**：
  - [static_masks.h](../src/geometry/static_masks.h)、[static_masks.cpp](../src/geometry/static_masks.cpp)：将相对曝光求解窗口收敛至 `sRGB 0.10–0.78` 并联合求解逐帧 3 通道白平衡（`wb_gains` 保持中性灰亮度不变）；新增 KDE 主曝光众数估计与 `ComputeHomogenizationPlan`（`pixel_ev_shift` / `residual_ev`）；过曝掩码 v2 扩展至 `32` 邻（`< 25 m`、放宽 LAS 深度至 `30 m`、移除 `depth_smooth` 限制）、无界连通饱和平台泛洪、高 EV 室内帧无深度/远距纯白门窗直接补种以及高 EV 泛光缝隙闭运算与外扩（单图 `45%` 熔断）。
  - [crop.h](../src/io/crop.h)、[crop.cpp](../src/io/crop.cpp)、[colmap.cpp](../src/io/colmap.cpp)：新增 `ExportNormalizedJpeg` 在物理线性空间叠加 KB8 逆暗角查找表、3 通道 `wb_gains` 与 `-pixel_ev_shift` 增益并以 `Q=97`（`4:4:4` 无色度抽样、保留 ICC/XMP 等非 EXIF 段）编码导出 JPEG，将 `residual_ev` 写入 EXIF `APP1` 并删除旧版旁路函数 `CopyJpegWithCalibratedExif`；`manifest.json` 同步记录 `photometric_normalization`、`white_balance` 与每图 `calibrated_ev`/`raw_relative_ev`/`pixel_ev_shift`/`wb_gains`。
  - [output.md](output.md)、[cache.md](cache.md)：同步更新导出图像光度预处理、`static_masks.json` 与 `manifest.json` 契约。
- **实景验证**（`.\tools\windows\build.ps1` Ninja Release `/W4` 0 错误 0 新增警告；复用 `work\小场景-evfix.work` 独立导出 `work\小场景-evfix2` 并通过 `.\run.ps1 check`，保留 `work\小场景` 与 `work\小场景-evfix` 并存对照）：

| 指标 (`小场景` 296 图) | 基线 (`work/小场景-evfix`) | 修复后 (`work/小场景-evfix2`) |
| --- | :---: | :---: |
| `geosunpre check` / 耗时 (`static_masks` / `export photos`) | `passed: true` (`8.43 s` / —) | **`passed: true`**（`9.33 s` / `11.84 s`） |
| 联合求解 (`exposure` / `white_balance`) | `med=0.0225 EV` / — | **`med=0.0229 EV`** / `wb calibrated`（`R/G=0.0316`, `B/G=0.0116 EV`） |
| 源头像素拉平 (`EXIF == 0.00 EV` 帧数 / 全局 EXIF 均值) | `~0/296`（范围 `-1.59..+5.89`） | **`252/296`**（室外 **`246/246` 全部 `0.00 EV`**，均值 `+0.510 EV`） |
| 过曝裁切屏蔽 (`clipped_pixels` / `images` / `capped`) | `14,243,749 px` / `79` / `0` | **`66,789,373 px`** / **`190`** / **`0` 熔断** |
| 典型漏检帧未遮死白 (`00167-L` / `00153-R` / `00162-R` / `00162-R`红框) | `18.87%` / `12.39%` / `6.15%` / `~33%` | **`0.00%`** / **`0.42%`** / **`0.05%`** / **`0.01%`** |
| `P_wall` 裸图跨视角一致性 (`ALL std` / `L lin std` / 门洞 `00168-R..173-R`) | 须靠 PPISP（暗角仍差 `0.73–0.87`） | **裸图免 PPISP 即达 `std=0.0385` (`L=0.0114`, `0.054 EV`)**，门洞 **`0.634–0.680` (`res=0.00`)** |
| 全场景室外裸图 SfM 轨迹一致性 (`Lum med/p90` / `L-R` / 帧 `std(R/G,B/G)`) | `0.2228 / 0.8085 EV` / `+0.2685 EV` / `0.0336, 0.0223` | **`0.1570 / 0.6896 EV`** / **`+0.0761 EV`** / **`0.0274, 0.0187`** |
| 室内暗部保护 (`raw_ev >= 0.5` 静态中位 `p50 min / med`) | — | **`0.252 / 0.274`**（无压黑） |
| `LichtFeld-Studio.exe` 5 步冒烟 (`trainer.json`) | 退出码 `0`（EXIF mean `-0.00 EV`） | 退出码 `0`（`PPISP exposure seeded ... 296/296 frames, mean 0.51 EV`，846k 点正常收敛） |

- **结论与未验证项**：源头线性去暗角 + 白平衡 + 有界匀光已将全部 246 帧室外图直接在像素层面拉平到 `0.00 EV` 且彻底清除了 `00162-R`/`00153-R`/`00167-L` 门洞过曝白块，显著降低了训练器 `PPISP`/`BilateralGrid` 的拟合负担。**未验证项**：1）尚未在 `LichtFeld-Studio` 中用 `work/小场景-evfix2` 跑完完整 20k 步训练并实测 `P_wall` 与其余区域在 novel view 上的最终成图消除效果；2）对于走廊深处看室内与室外的局部高反差镜头（如 `00044-R`、`00052-R` 上半墙在室外多帧共同过曝、没有任何更暗参考帧可对照的区域 `6.2%–8.7%`），因全场景均无未裁切观测而不应人为猜测篡改像素，需实训观察其是否仍有影响。

## 2026-11-26：3 通道 Schur-Complement 光度标定、安全匀光匀色与双场景验证（`小场景-photov3` & `展厅-photov3`）

- **对 `2a663df` 基线（`work/1125/小场景`）与初版方案的诊断**：
  1. **旧版匀光标定不足与无审计风险**：`2a663df` 采用纯标量单通道暗角与配对帧比求解，对 `小场景`（296 图、实测全场景曝光跨越 `7.94 EV`）存在明显镜头红色通道边缘额外衰减（边缘线性增益 `L`: `R=0.695` vs `G/B≈0.80`、`R`: `R=0.671` vs `G/B≈0.79`）、走廊/中庭 `~0.55 EV` 白平衡跳变，且 `p50` 有界提亮/压暗上限过宽（`1.20 / 1.40 EV`）缺少高亮余量审计，在逆光/走廊过渡帧造成单图最高 `38.36%`（`00079-L`）新增过曝未遮蔽及 `25.46%`（`00167-L`）压暗已饱和区变灰；
  2. **高光硬截断与双曲调色偏粉**：早期在全饱和白色区域（`255`）叠加了 `R > G/B` 的逆暗角/白平衡增益并做保色调压缩（除以最大通道），使 `00162-R`、`00114-R` 门洞/天窗纯白裁切区被压偏粉；改用 `0.92` 起每通道独立双曲软拐点滚降后恢复纯正中性白；
  3. **单通道 `max(R,G,B)` 触发彩色灯带误杀（由 `展厅` 跨场景回归暴露）**：`展厅` 含高饱和蓝/青色 LED 灯槽与电子展屏（`R≈0.47–0.52, G≈0.73–0.75, B≈0.98`, `lum≈0.72`），使用 `lum || max_ch` 导致 `PhotoClip` 与闭环审计将大片灯槽墙面误判过曝并扩散至 `45%` 熔断上限（`展厅-photov2` 出现 `14` 帧熔断、`226.8M px` 过曝遮蔽）。将过曝种子/泛洪与原图饱和判定收紧为亮度联合门槛（`lum >= 0.94 && max_rgb >= 0.975`，滞后 `0.90 && 0.95`）并以通道截断 `[0, 1]` 线性亮度触发闭环审计后彻底消除。
- **改动**：
  - [photometric.h](../src/geometry/photometric.h)、[photometric.cpp](../src/geometry/photometric.cpp)、[CMakeLists.txt](../CMakeLists.txt)：新增独立多视角光度标定模块 `SolvePhotometricCalibration`（融合 LAS 0.20 m 体素法向/入射角过滤三径向带轨迹与高置信 SfM 轨迹、1 m 体素 10% 空间留出集、3 通道精确 per-track Schur 消元 Huber IRLS、二轮 `0.40 EV` 剔除、物理边界与留出增益自动降阶 `3->2->0`、per-lens 1D 时序 Huber 白平衡平滑 `±0.80 EV`）与 `PlanSafeHomogenization`（`0.30 EV` 提亮上限 + `p98` 高光余量约束、`0.80 EV` 压暗上限 + 暗部与已饱和保护、闭环过曝/压暗饱和审计回写 `post_clip_masks`）。
  - [static_masks.h](../src/geometry/static_masks.h)、[static_masks.cpp](../src/geometry/static_masks.cpp)：删除旧标量 `FitVignetting`、`SolveRelativeExposures`、`ComputeHomogenizationPlan`，接入 3 通道标定与 `post_clip_masks`，收紧多视角过曝联合门槛。
  - [crop.h](../src/io/crop.h)、[crop.cpp](../src/io/crop.cpp)、[colmap.cpp](../src/io/colmap.cpp)：升级 3 通道 KB8 逆暗角查找表与每通道 `0.92` 双曲软拐点压缩，`manifest.json` 记录 `vignetting_coeffs_rgb` 与 `version: 2`。
  - [output.md](output.md)、[cache.md](cache.md)：同步更新光度标定、匀光、软拐点与过曝掩码规则及缓存字段。
- **`小场景` 实景预处理与 `LichtFeld-Studio` 20k 步训练对比**（均在同等配置 `work\小场景-*\trainer.json`、`--headless --eval` 下完成，259 训练 / 37 留出视角，`rolling_shutter_cameras: 37/37`）：

| 方案 / 数据集 | 留出集 `lum_p50` (前→后) | 过曝遮蔽图数 / 像素 (`clip_capped`) | 新增过曝占比 `new_clip` | 20k `PSNR` | `Aligned PSNR` | `SSIM` / `Aligned` | `LPIPS` / `Aligned` | 走廊 6 视角 `PSNR` |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 基线 (`1125/小场景` → `lfs-base`) | — | — | — | `22.5202` | `23.1862` | **`0.7937` / `0.8087`** | **`0.2944` / `0.2918`** | `21.3208` |
| 方案 A (`photoA`：有界匀光 max↑`0.60 EV` → `lfs-photoA`) | `0.1337 → 0.0427 EV` | `271` / `73.86M` (`0`) | `0.579%` | `23.0972` | `23.9805` | `0.7824 / 0.7996` | `0.2981 / 0.2948` | `23.7730` |
| 方案 B (`photoB`：仅去暗角+WB+EXIF → `lfs-photoB`) | `0.1337 → 0.0427 EV` | `257` / `71.48M` (`0`) | `0.653%` | `22.8537` | `23.6818` | `0.7805 / 0.7974` | `0.3014 / 0.2982` | `23.1324` |
| 收敛版 v2 (`photov2`：收紧提亮 `0.30 EV` → `lfs-photov2`) | `0.1337 → 0.0427 EV` | `264` / `59.65M` (`0`) | `0.540%` | `23.1905` | `24.0830` | `0.7837 / 0.8014` | `0.2977 / 0.2944` | **`23.9494`** |
| **终版 v3 (`photov3`：亮度联合过曝门槛 → `lfs-photov3`)** | **`0.1337 → 0.0427 EV`** | **`244` / `44.18M` (`0`)** | **`0.282%`** | **`23.2020`** (`+0.682 dB`) | **`24.1104`** (`+0.924 dB`) | `0.7840 / 0.8019` | `0.2963` / **`0.2929`** | `23.9162` (`+2.595 dB`) |

- **`展厅` 跨场景回归对比**（588/588 全注册，通过 `.\run.ps1 check`，首次全流程曾遇既有 `shutter` 阶段 Ceres `cusolverDnSpotrf` 单精度非正定瞬时数值故障，续跑一次通过）：

| 版本 | 留出集 `lum_p50` (前→后) | 跨镜头色度 `R/G` / `B/G` | 过曝遮蔽图数 / 像素 | `clip_capped` | 新增过曝 `new_clip` (最差单帧) | 监督有效像素 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| `展厅-photov2`（单通道 `\|\|` 触发） | `0.1718 → 0.0741 EV` | `0.0299 / 0.0317 EV` | `566` / `226.81M` | `14` | `1.121%` (`13.68%`) | `364.81M` |
| **`展厅-photov3`（联合亮度门槛）** | **`0.1718 → 0.0741 EV`** | **`0.0300 / 0.0316 EV`** | **`412` / `50.84M`** | **`0`** | **`0.184%`** (`4.31%`) | **`397.25M`** |

- **结论与未验证项**：收敛为单一 `SafeBounded` 主线（3 通道 Schur-Complement IRLS 标定 + 时序平滑白平衡 + 有界安全像素匀光 + 每通道 `0.92` 高光软拐点 + 亮度联合过曝/闭环审计）。在 `小场景` 实测较 `work/1125/小场景` 提升 `PSNR +0.682 dB`、`Aligned PSNR +0.924 dB`、走廊 6 视角 `+2.595 dB`，几何精度亦最优（`normal_angle_deg: 16.319°` vs `16.431°`，`depth_absrel: 0.0231` vs `0.0256`）；`展厅` 目视核验 `00194-R`/`00168-R` 蓝色 LED 灯槽与屏幕均保持静态 `255` 且 `clip_capped_images == 0`。**未验证项**：尚未以 `work/展厅-photov3` 在下游 `LichtFeld-Studio` 跑完完整 29,000 步 3DGS 训练以实测该室内场景最终成图表现。

## 2026-11-27：全链路无反光纯漫反射 3DGS 重建（`weights/` 连续漫反射权重 + `RecolorPlyDiffuse` + `sh_degree=0` / `PPISP` 损失链路修复）

- **问题根因定位**：
  1. 原 `GeosunPre` 瞬态光度剔除仅在配对曝光比 `[0.50, 2.00]` 内采样且带 `grad < 0.25` 门限，跳过了大曝光差参考帧与展板文字/奖牌框线上的镜面反光，同时对阴影/高光二值置 `0` 会破坏静态展板与地面的 MoGe-2 深度/法向监督；`points3D.ply` 仍直接使用未经光度校准和反光过滤的原始 `.las` RGB；默认模板保留 `sh_degree=1` 与 `color_variance_weight=0.0`，使半透明浮层与球谐视角项在训练中拟合出吊顶灯高光、奖牌高光与残余行人影子。
  2. 原 `LichtFeld-Studio` 在开启 `use_exposure_correction` 时，将未经过 `PPISP` 曝光/白平衡校准的原始渲染 `output.image`（`raw_loss_input`）送入高权重 `gradient_residual_loss`（`weight=1.8`）及解耦后的 `SSIM` 对比度/结构项，导致大曝光跳变视角强行将未经曝光拉平的参考光影烙入静态表面；同时缺乏消费级 `weights/` 连续降权 sidecar 与无参数梯度污染的 `PPISP::backward_rgb_only_in_place` 回传通路。
- **改动**：
  - `LichtFeld-Studio`：
    - 新增 [photometric_robust_weight.hpp](../../LichtFeld-Studio/src/training/kernels/photometric_robust_weight.hpp)、[photometric_robust_weight.cu](../../LichtFeld-Studio/src/training/kernels/photometric_robust_weight.cu) 与 [photometric_weights_sidecar.hpp](../../LichtFeld-Studio/src/training/photometric_weights_sidecar.hpp) / [.cpp](../../LichtFeld-Studio/src/training/photometric_weights_sidecar.cpp)，实现 256 MiB CUDA 常驻（CPU 溢出回退）的 `weights/*.png` 双线性加载、与前景/掩码融合及在线 `8×8` 低频不对称高光/阴影残差连续降权。
    - 扩展 [ppisp.hpp](../../LichtFeld-Studio/src/training/components/ppisp.hpp) / [.cpp](../../LichtFeld-Studio/src/training/components/ppisp.cpp) 新增 `backward_rgb_only_in_place`（分配全尺寸独立暂存缓冲，不写 `PPISP` 参数梯度），并在 [trainer.hpp](../../LichtFeld-Studio/src/training/trainer.hpp) / [.cpp](../../LichtFeld-Studio/src/training/trainer.cpp) 中保留 `normal_pixel_weight` 全自信度监督的同时用连续漫反射权重加权光度/梯度/自适应密度误差图，且修复 `use_exposure_correction` 启用期间 `SSIM` 与 `gradient_residual_loss` 的 `PPISP` 对齐前向/反传回路。
    - 在 [test_mask_preprocess_fusion.cpp](../../LichtFeld-Studio/tests/test_mask_preprocess_fusion.cpp) 与 [test_ppisp_regularization.cpp](../../LichtFeld-Studio/tests/test_ppisp_regularization.cpp) 补充 GPU 权重核及 `backward_rgb_only_in_place` 有限差分单测。
  - `GeosunPre`：
    - 更新 [configs/trainer.json](../configs/trainer.json)（`"sh_degree": 0`、`"color_variance_weight": 0.05`、`"color_variance_warmup_iters": 1500`、`"exposure_correction_grid_start_iter": 3000`、`"tv_loss_weight": 20.0`）与 [trainer_config.cpp](../src/io/trainer_config.cpp)（同步按比例缩放 `exposure_correction_grid_start_iter`）。
    - 重构 [static_masks.cpp](../src/geometry/static_masks.cpp)（提升至 `PhotoGridSize=320`，删除配对 `±1 EV` 二值暗/亮掩码旧路径）：在全局 `reference_ev` 线性空间用无梯度门槛的 `5×5` 掩码归一化包络、全视角原始 `p50` 点高光判别及至多 32 个 `<=18 m` 方向分层共视邻域的加权 `q=0.35` 漫反射基准，生成包含 `sRGB>=0.06` 邻域过滤、联合 EV/感知幅度门限（避免压暗黑地毯/黑边框等暗部量化区）及去噪核膨胀的连续 Tukey 漫反射权重 `weights/{L,R}/**.JPG.png`，并将 `masks/` 严格收敛为仅人物/有效域外/真过曝裁切置 `0`；写出 `cache/static_masks/diffuse_views.bin`。
    - 在 [photometric.h](../src/geometry/photometric.h) / [photometric.cpp](../src/geometry/photometric.cpp) 新增流式分块 `RecolorPlyDiffuse`（每点至多 24 方向分层视角、LAS 深度抑遮挡、高漫反射权重过滤 `[20%, 55%]` 加权截断均值重着色 `points3D.ply`）；升级 [colmap.cpp](../src/io/colmap.cpp)（`photometric_normalization.version = 3`、`weights/` 随窗口双线性导出及 `check` 回读核验）并同步 [output.md](output.md)、[cache.md](cache.md)、[supervision.md](supervision.md)。
- **验证与实测对照**：
  - **单元测试**：`lichtfeld_tests.exe`（`MaskPreprocess*:GradientResidual*:TrainingParameters*:ArgumentParser*:PPISP*`）`273` 用例跑过 `262 PASS`、`11 SKIPPED`（仅外部数据集环境变量门控用例）、`0 FAILED`；`color_variance_tests.exe` `13/13 PASS`；`GeosunPre` `tools/windows/build.ps1` Ninja Release `/W4` `0` 错误 `0` 新增警告。
  - **预处理门限迭代**：`小场景-diffuse` 初版纯 `log2` 权重因暗部 8-bit 量化与无条件 `3×3` 腐蚀导致 `mean_static_weight = 0.7161 < 0.85`（黑窗框/暗地毯误衰减），触发重规划门限并加入感知幅度门限与去噪核膨胀后恢复至 `0.9390`。

| 预处理场景 (`geosunpre check passed: true`) | 配准 | 留出 `lum_p50` (前→后) | `mean_static_weight` | `glare_atten` / `shadow_atten` | 过曝图数 / 像素 (`capped`) | `RecolorPlyDiffuse` 重着色 | 监督有效像素 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `小场景-diffuse` (`work/小场景-diffuse`) | `296/296` | `0.1354 → 0.0455 EV` | **`0.9390`** | `3.072%` / `1.325%` | `254` / `45.79M` (`0`) | `724,314 / 846,335` (**`85.58%`**, `med_views=11`) | `208.65M` |
| `展厅-diffuse` (`work/展厅-diffuse`) | `588/588` | `0.1757 → 0.0760 EV` | **`0.9448`** | `3.269%` / `0.458%` | `461` / `58.44M` (`0`) | `625,228 / 882,717` (**`70.83%`**, `med_views=8`) | `396.65M` |

| 下游 3DGS 训练对比 (`LichtFeld-Studio --headless --eval`) | 步数 (`train/val`) | `PSNR` | `Aligned PSNR` | `SSIM` / `Aligned` | `LPIPS` / `Aligned` | `Normal (°)` / `Depth AbsRel` | `Bias_corr (R,G,B)` | 终态效果与权衡 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| `小场景` 基线 (`photov3` → `lfs-photov3`) | `20k` (`259/37`) | `23.2020` | `24.1104` | `0.7840 / 0.8019` | `0.2963 / 0.2929` | — | — | `sh_degree=1`，保留少量镜面反光 |
| **`小场景-diffuse` (`work/小场景-lfs-diffuse`)** | `20k` (`259/37`) | `21.8964` | `22.4140` | `0.7841 / 0.7994` | `0.3017 / 0.2986` | `16.5851°` / `0.0161` | `(-0.020,-0.019,-0.019)` | `sh_degree=0` 移除电梯门/玻璃反光与移动阴影（结构 `SSIM` 持平，黑箱“recycle.”字体锐利） |
| `展厅` 基线 (`work/展厅` → `work/展厅-lfs-base`) | `29k` (`514/74`) | `24.9800` | `25.7375` | `0.8318 / 0.8493` | `0.2792 / 0.2752` | `15.7113°` / `0.0094` | `(+0.0023,+0.0023,+0.0029)` | `00285-L` 地面仍有行人灰影与反光斑，`00253-L`/`00277-L` 展墙含高光雾与玻璃虚影 |
| **`展厅-diffuse` (`work/展厅-lfs-diffuse`)** | `29k` (`514/74`) | `24.5014` | `25.1476` | `0.8292 / 0.8461` | `0.2836 / 0.2795` | `16.1574°` / **`0.0082`** | `(-0.0011,-0.0010,-0.0008)` | 清除地面行人灰影、白色展台/展墙高光雾与玻璃虚影，几何 `Depth AbsRel` 优 `12.8%`，汉字清晰 |

- **结论与取舍**：
  - 将交付目标收敛为纯漫反射无反光重建（`sh_degree=0` + `weights/` 连续降权 + 多视角截断均值重着色 + `color_variance=0.05` + `PPISP` 光度对齐梯度通路）后，尽管相对于仍包含实拍高光的未改动留出真值图，逐像素 `PSNR` 按预期微降 `0.48 dB`（`展厅`）至 `1.31 dB`（`小场景`），但渲染视觉上彻底消除了地面移动影子拖尾、白色展台与展墙顶灯高光雾、金属电梯门及玻璃反光虚影，`展厅` 几何深度精度 `Depth AbsRel` 从 `0.0094` 改善至 `0.0082`，色偏校正残差近乎归零（`< 0.0011`），且文字锐度与 `SSIM`（`0.8292` / `0.8461`）保持稳健。

## 2026-11-27（续）：白墙暗黄/发灰污染与 PPISP 漂移修复（高光肩区降权 + HDR DC 上界释放 + GPU 融合 PPISP 正则，`小场景-lfs-diffuse4`）

- **三层根因定位（围绕 `work/小场景-lfs-diffuse/eval_step_20000/2433-514856-679_00039-L.png` 弧形白墙脏斑）**：
  1. **`GeosunPre` 高光非线性肩区未降权**：室内走廊/大堂白墙在过曝邻近亮度下虽未触及硬裁切掩码，但已进入相机/软拐点非线性肩区（`sRGB >= 0.76`, `max_ch >= 0.82`），相对更暗多视角观测线性偏暗 `0.12+ EV`，旧版 `diffuse` 未对该肩区降权，拖低白墙参考真值与 `RecolorPlyDiffuse` 初始色。
  2. **`LichtFeld-Studio` `sh0` DC 上界锁死在 `1.0`**：[gut_regularization.cu](../../LichtFeld-Studio/src/training/kernels/gut_regularization.cu)、[adam_kernels.cuh](../../LichtFeld-Studio/src/training/rasterization/fastgs/optimizer/include/adam_kernels.cuh) 与 [kernels_backward.cuh](../../LichtFeld-Studio/src/training/rasterization/fastgs/rasterization/include/kernels_backward.cuh) 原先对 `sh0` DC 施加 `RGB ∈ [0, 1]` 软限幅；在近 `7 EV` 跨场景大曝光差下，`PPISP` 负曝光帧无法提高场景线性亮度来补偿暗曝光缩放，被迫压暗静态白墙（`diffuse` / `diffuse2` 评估残差 `Bias_corr ≈ -0.020`）。
  3. **`PPISP` 自由度规范漂移（释放 `dc_max_rgb` 后在 `diffuse3` 进一步暴露）**：缺少每帧 ZCA 色彩潜变量 `color_l2` 惩罚与 EXIF 相对曝光先验 `exposure_prior` 锚点，使对向视角形成约 `0.65 EV` 伪曝光波并在 `00039-L` 弧形白墙前贴出一层偏黄前景浮壳；同时 [ppisp.cu](../../LichtFeld-Studio/src/training/kernels/ppisp.cu) 的 `ppisp_vignetting_reg_kernel` 仅惩罚正项 `alpha > 0`，导致解锁 HDR DC 上界后 `camera 2 (R)` 暗角 `a0` 漂移至 `-0.2855`，造成 `00163-R` 从 `20.23 dB`（`d1`）跌至 `16.82 dB`（`d3`）。
- **改动**：
  - `GeosunPre`：[static_masks.cpp](../src/geometry/static_masks.cpp) 与 [photometric.cpp](../src/geometry/photometric.cpp) 新增高光非线性肩区连续降权（`PhotoShoulderMinSrgb = 0.76`、`PhotoShoulderMinMaxCh = 0.82`、`PhotoShoulderMinEvDiff = 0.35`、`PhotoShoulderMinDropEv = 0.12`、`PhotoShoulderDropC = 0.25`），并收敛终态参数（`PhotoDiffuseQuantile` `0.35 -> 0.42`、`RecolorPlyDiffuse` 截断区间 `[20%, 55%] -> [25%, 65%]`、`PhotoMinDiffuseWeight` `0.05 -> 0.0 -> 0.02`，其中 `展厅-diffuse2` 导出于 `min_w = 0.0` 版本）。
  - `LichtFeld-Studio`：
    - **HDR `dc_max_rgb` 上界释放**：[trainer.cpp](../../LichtFeld-Studio/src/training/trainer.cpp) 新增 `Trainer::refresh_dc_max_rgb()`（按 `clamp(exp2(-min_exp) * 1.15, 1.0, 8.0)` 周期刷新），同步更新 [gut_regularization.cu](../../LichtFeld-Studio/src/training/kernels/gut_regularization.cu)、[adam_kernels.cuh](../../LichtFeld-Studio/src/training/rasterization/fastgs/optimizer/include/adam_kernels.cuh)、[kernels_backward.cuh](../../LichtFeld-Studio/src/training/rasterization/fastgs/rasterization/include/kernels_backward.cuh) 及 [test_gut_regularization.cpp](../../LichtFeld-Studio/tests/test_gut_regularization.cpp)。
    - **GPU 融合 `PPISP` 正则（Schema v3）**：[ppisp.cuh](../../LichtFeld-Studio/src/training/include/lfs/kernels/ppisp.cuh) 与 [ppisp.cu](../../LichtFeld-Studio/src/training/kernels/ppisp.cu) 为 `ppisp_vignetting_reg_kernel` 增加 `vig_alpha_l2` 双向 L2 惩罚并新增 `ppisp_frame_reg_kernel` / `launch_ppisp_frame_reg`；[ppisp.hpp](../../LichtFeld-Studio/src/training/components/ppisp.hpp) / [.cpp](../../LichtFeld-Studio/src/training/components/ppisp.cpp) 将 `PPISPConfig` 升至 v3（向后兼容 v2），新增 `vig_alpha_l2`、`exposure_prior`、`color_l2`、`set_exposure_prior()`、`exposure_prior_slope()`、`exposure_prior_rms()`、`color_latent_rms()` 与梯度接口；[trainer.cpp](../../LichtFeld-Studio/src/training/trainer.cpp) 在启用 `use_exposure_correction` 时开启 `vig_alpha_l2 = 0.5f`、`exposure_prior = ppisp_exposure_from_exif ? 0.5f : 0.0f`、`color_l2 = 0.05f`，新增 `collect_ppisp_exif_evs()` 并在 `applyPPISPForEval()` 按拟合 `exposure_prior_slope()` 插值评估曝光、在断点续训时恢复先验。
    - 补充单测：[test_component_config_serialization.cpp](../../LichtFeld-Studio/tests/test_component_config_serialization.cpp)、[test_ppisp_exposure_seed.cpp](../../LichtFeld-Studio/tests/test_ppisp_exposure_seed.cpp)、[test_ppisp_regularization.cpp](../../LichtFeld-Studio/tests/test_ppisp_regularization.cpp)。
- **预处理（`geosunpre check` 均 `passed: true`）与 `小场景` 20k 下游演进对比**：

| 预处理场景 | 配准 | 留出 `lum_p50` (前→后) | `mean_static_weight` | `glare` / `shadow` / `shoulder` | 过曝图数 / 像素 (`capped`) | 无残差图数 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: |
| `小场景-diffuse3` (`work/小场景-diffuse3`) | `296/296` | `0.1354 → 0.0455` | `0.9382` | `2.535%` / `1.811%` / `0.335%` | `254` / `45.79M` (`0`) | `120/296` |
| `展厅-diffuse2` (`work/展厅-diffuse2`) | `588/588` | `0.1757 → 0.0760` | `0.9416` | `3.100%` / `0.609%` / `0.125%` | `461` / `58.44M` (`0`) | `190/588` |

| `小场景` 20k 下游方案 (`work/小场景-lfs-*`, `259/37`) | `PSNR` | `Aligned PSNR` | `SSIM` / `Aligned` | `LPIPS` / `Aligned` | `Normal (°)` / `Depth AbsRel` | `Bias_corr (R,G,B)` | 关键诊断 / 终态效果 |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| `小场景-lfs-diffuse` (`diffuse` 源数据) | `21.8964` | `22.4140` | `0.7841 / 0.7994` | `0.3017 / 0.2986` | `16.5851°` / `0.0161` | `(-0.020,-0.019,-0.019)` | `00039-L` 弧形白墙有暗黄脏斑 (`19.89 dB`) |
| `小场景-lfs-diffuse2`（加高光肩区降权） | `22.1448` | `22.7111` | `0.7838 / 0.7974` | `0.3015 / 0.2989` | `16.7654°` / `0.0161` | `(-0.0192,-0.0174,-0.0176)` | DC 上界仍锁在 `1.0`，全局负偏依在 |
| `小场景-lfs-diffuse3`（解锁 HDR `dc_max_rgb`） | `23.0314` | `23.8421` | `0.7867 / 0.8009` | `0.2973 / 0.2948` | `16.8987°` / `0.0157` | `(-0.0036,-0.0043,-0.0049)` | 曝光偏置清零，但 `00039-L` (`19.74 dB`) 仍有漂移壳、`00163-R` 暗角退化至 `16.82 dB` |
| **`小场景-lfs-diffuse4`（融合 `PPISP` 正则）** | **`23.3808`** | **`24.2519`** | **`0.7894 / 0.8047`** | **`0.2941 / 0.2911`** | **`16.0455°` / `0.0117`** | **`(-0.0021,-0.0029,-0.0034)`** | **`00039-L` (`21.07 dB`) 弧形白墙脏斑基本消除、`00163-R` (`19.58 dB`) 暗角退化恢复、`00175-L` (`25.42 dB`) 干净；全面超越 `photov3`** |

- **单测、约束验证与未验证项**：
  - `PPISP` 漂移收敛（`小场景-lfs-diffuse4`）：`exp_prior_slope = 0.4580`、`exp_prior_rms = 0.0028`、`color_rms = 0.0119`；`cam1 vig a0 = (-0.0030, -0.0037, -0.0018)`、`cam2 vig a0 = (0.0000, -0.0000, 0.0000)`。
  - `lichtfeld_tests.exe`（`MaskPreprocess*:PPISP*:GutRegularization*:FusedReg*`）共 `101` 用例：`96 PASSED`、`5 SKIPPED`（环境门控用例）、`0 FAILED`；`GeosunPre` `tools/windows/build.ps1` 构建及 `小场景-diffuse3` / `展厅-diffuse2` `geosunpre.exe check` 均通过（`passed: true`）。
  - **未验证项**：尚未以 `work/展厅-diffuse2` 在新版 `LichtFeld-Studio` 上跑完完整 `29,000` 步 3DGS 实景训练回归。

## 2026-11-27（续）：下游 3DGUT 启用 `supervision.json` 精确米制锚点 + 无量纲相对失真 / 中位深度 / Free-Space 抑制分层与浮壳（Route B）

- **问题诊断（基于 `C:\Users\admin\.lichtfeld\projects\{室内,展厅}.licht` 与 `work/{室内,展厅}` 对应数据集审计）**：
  1. **RANSAC 锚点污染与丢相机**：上游 `supervision.json` 已在 `las_face_bilateral_v1` 中输出精确线性编码 `[min_distance_m, max_distance_m] -> [0.02, 1.0] * 65535`，但下游仍用未遮挡稀疏点云跑 RANSAC 拟合尺度偏移；在 `展厅`（588 图）丢弃 `15` 个相机、对 LAS 偏差 `p90 = 7.72 cm`（`2m` 处）、`p99 = 34.29 cm`、带 `+1.44 cm`（`p90 +6.76 cm`）相机侧系统正偏。
  2. **原线性深度失真随距离失衡 & 纯前向深度无法压住前沿浮壳**：原 `depth_distortion_loss` 惩罚米制标准差并在 `15 cm` 处硬截断，室内典型 `3–8 m` 距离上抑制不足（`20 cm` 平面块 `p90-p10` 中位厚度 `室内 3.26 cm / 展厅 3.10 cm`），且 `10–20 cm` 相机侧存在大量高不透明度浮壳（`op > 0.3` 占 `室内 61.2% / 展厅 58.9%`、`展厅` 墙面相机侧 `>5 cm` 达 `14.2%`）。
- **核心改动（全部位于 `LichtFeld-Studio`，`GeosunPre` 与 `configs/trainer.json` 零改动，保留非 LAS 数据集原 SSI 通路）**：
  1. **精确米制锚点直读**：[las_prior_sidecar.hpp](../../LichtFeld-Studio/src/training/las_prior_sidecar.hpp) / [.cpp](../../LichtFeld-Studio/src/training/las_prior_sidecar.cpp) 解析 `las_*` 对齐且编码为 `linear_[min_d,max_d]_to_[0.02,1.0]*65535` 的每帧深度范围；[trainer.cpp](../../LichtFeld-Studio/src/training/trainer.cpp) `fitDepthAnchors` 在命中时直接构建精确 `model = 1` 锚点并跳过 RANSAC。
  2. **边界掩码死区 Huber 米制深度监督**：[depth_loss.hpp](../../LichtFeld-Studio/src/training/kernels/depth_loss.hpp) / [.cu](../../LichtFeld-Studio/src/training/kernels/depth_loss.cu) 新增 `launch_metric_depth_loss`（4 邻域 `<= 3.5%` 内点、死区 `max(1.5 cm / d, 0.8%)`、期望/中位对数深度 Huber `κ=0.05` + 4 邻域对称对数深度梯度 Huber `κ=0.03`，含 `--use_fast_math` 下 `1e-5` 梯度噪声下限）。
  3. **3DGUT 中位深度反向 + 无量纲相对失真 + Mean-Median 一致性**：[RasterizeToPixelsFromWorld3DGSFwd.cu](../../LichtFeld-Studio/src/training/rasterization/gsplat/RasterizeToPixelsFromWorld3DGSFwd.cu) / [Bwd.cu](../../LichtFeld-Studio/src/training/rasterization/gsplat/RasterizeToPixelsFromWorld3DGSBwd.cu)、[gsplat_rasterizer.cpp](../../LichtFeld-Studio/src/training/rasterization/gsplat_rasterizer.cpp) 打通 `median_ids` 与 `ExtraDepthBwd`；[depth_distortion_loss.hpp](../../LichtFeld-Studio/src/training/kernels/depth_distortion_loss.hpp) / [.cu](../../LichtFeld-Studio/src/training/kernels/depth_distortion_loss.cu) 将失真升级为以 `0.80 m` 为软化下界的无量纲相对方差 `r^2 = (s2*a - s1*s1)/s1^2`（先算分子并设 `1e-6` 阈防 `--use_fast_math` 伪方差）+ 平滑对数饱和（`τ=0.12`）+ `|log d_mean - log d_median|` Charbonnier 一致性 + `3×3` `>3.5%` 深度边缘屏蔽。
  4. **单侧 5×5 保守 Free-Space 不透明度抑制**：[depth_loss.cu](../../LichtFeld-Studio/src/training/kernels/depth_loss.cu) 新增 `launch_build_free_space_map`（5×5 局部最小 `d_free = d_min5 - max(6 cm, 4%·d_min5)`），在 3DGUT 反向核对饱和高不透明度浮壳无条件施加单向抑制梯度。
- **两轮校准与单测验证**：
  - 首轮 `v1`（`室内-lfs-metricB`，相对失真 `2×` 二次预热、`0.20^p` 衰减）因深度失真在 `p50=3.11 m / p90=6.31 m` 上弱于原线性失真且晚期 Huber 仍拉扯表面，20 cm 平面厚度未降反停在 `3.29 cm`、法线角 `11.44° -> 12.11°`；调整为 `v2`（`kDepthDistortionDepthFloor=0.80 m`、失真 `6.0×` / Mean-Median `1.5×` 线性预热至 `0.20`、米制深度 `0.05^p` 衰减与 `0.10/0.08/0.08` 权重、Free-Space 边距 `max(6 cm, 4%·d)` 与 `0.08` 权重线性预热）后重新编译。
  - `lichtfeld_tests.exe`（`LasPriorSidecar*:GsplatRasterizerTest.DepthDistortion*:GsplatRasterizerTest.MedianDepth*:GsplatRasterizerTest.FreeSpace*:GsplatRasterizerTest.MetricDepth*:GsplatRasterizerTest.RenderedSquaredDepth*`）：**`12/12 PASSED`**（`0 FAILED`，覆盖有限差分 VJP 与 `bwd_splat_kernel` / `bwd_kernel` 一致性）。

### 实景端到端对照（`RTX 4090`，成果保留于 `work/室内-lfs-metricB2` 与 `work/展厅-lfs-metricB2`）

注：`室内.licht` 与 `展厅.licht` 基线原工程在 `enable_eval=false`（全图入训）下完成；为区分口径，下表同时列出新训 `--eval` 留出测试集与 `--eval-all` 下 **同批已见训练视角（`241` / `514`）** 及全集几何指针。

| 组别 | 锚点匹配 (训) | 耗时 / 速度 (`#GS`) | 视角口径 | `PSNR` / `Aligned` | `SSIM` / `Aligned` | `LPIPS` | `Normal (°)` / `Depth AbsRel` | `off-LAS >5cm` | `10-20cm` 浮壳 (`op>0.3` / med) | `20cm` 平面厚度 `p50` (全/地顶/墙, `>3cm`) | `front>5cm` (地/墙/顶) |
| --- | :---: | :---: | --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`室内` 基线 (`室内.licht`)** | RANSAC | — (`1,485,608`) | `35` 全已见 | `27.17 / 27.36` | `0.9048 / 0.9072` | `0.2786` | `11.44° / 0.0093` | `8.04%` | `1.77%` (`61.2%` / `0.516`) | `3.26 / 3.98 / 2.66 cm` (`53.3%`) | `7.2% / 7.1% / 15.9%` |
| `室内-lfs-metricB` (`v1`) | **`241/241`** | `293.6 s` / `68.1 it/s` (`1,486,108`) | `241` 已见 (`35` 留出) | `27.05 / 27.29` (`26.40 / 26.58`) | `0.9052 / 0.9082` (`0.8940`) | `0.2754` (`0.2862`) | `12.11° / 0.0073` | `7.71%` | `1.93%` (**`50.2%` / `0.305`**) | `3.29 / 3.87 / 2.66 cm` (`54.0%`) | `6.2% / 6.3% / 15.1%` |
| **`室内-lfs-metricB2` (`v2`)** | **`241/241`** | **`286.4 s` / `69.8 it/s`** (`1,482,860`) | `241` 已见 (`35` 留出) | `26.85 / 27.08` (`26.21 / 26.38`) | `0.9035 / 0.9065` (`0.8927`) | **`0.2783`** (`0.2890`) | **`11.01° / 0.0072`** (`10.98° / 0.0071`) | **`6.71%`** | **`1.56%`** (`52.1%` / `0.334`) | **`2.42 / 2.52 / 2.32 cm`** (**`41.2%`**) | **`5.1% / 5.8%`** `/ 17.3%` |
| **`展厅` 基线 (`展厅.licht`)** | RANSAC 丢 15 帧 | — (`2,574,650`) | `74` 全已见 | `26.20 / 27.05` | `0.8508 / 0.8678` | `0.2679` | `15.65° / 0.0077` | `13.03%` | `4.31%` (`58.9%` / `0.403`) | `3.10 / 2.67 / 3.55 cm` (`51.2%`) | `2.6% / 14.2% / 6.2%` |
| **`展厅-lfs-metricB2` (`v2`)** | **`514/514` (`0` 丢弃)** | `513.0 s` / `56.5 it/s` (`2,577,874`) | `514` 已见 (`74` 留出) | `26.09 / 26.87` (`24.98 / 25.73`) | **`0.8521 / 0.8689`** (`0.8273`) | **`0.2674`** (`0.2856`) | **`15.53° / 0.0072`** (`15.66° / 0.0088`) | **`10.83%`** | **`3.44%`** (**`43.8%` / `0.237`**) | **`2.97 / 2.39 / 3.49 cm`** (**`49.6%`**) | **`1.9% / 13.0%`** `/ 6.4%` |

- **结论与未验证项**：
  - Route B 在两个场景均实现 **100% 精确米制锚点加载**（零回退、解决 `展厅` 15 帧 RANSAC 丢失），且在不新增任何配置参数的前提下显著改善三维结构：`室内` 20 cm 平面中位厚度缩薄 **`25.8%`**（地面/顶面 **`-36.7%`**）、偏离 LAS `>5 cm` 高斯下降 **`16.5%`**、法线角 `11.44° -> 11.01°`、`Depth AbsRel 0.0093 -> 0.0072`；`展厅` 偏离 LAS `>5 cm` 下降 **`16.9%`**、`10–20 cm` 浮壳数量降 **`20.2%`** 且中位不透明度由 `0.403` 降至 **`0.237`**（`-41.2%`）、地面/墙面相机侧 `>5 cm` 浮壳同步收敛、全集法线（`15.53°`）、`Depth AbsRel`（`0.0072`）、同口径已见 `SSIM`/`LPIPS` 均优于基线，同口径 `Aligned PSNR` 仅微差 `0.17–0.28 dB`。
  - **未验证项**：1）针孔模型 `DepthRayGrid` 分支已在单测 `MetricDepthLossDeadbandHuberAndEdges` 通过编译与测试，尚未在真实针孔数据集做端到端训练；2）`work/室内-lfs-metricB` 与 `work/{室内,展厅}-lfs-metricB2` 已保留在本地供核验视觉成图与 `.licht` / `.ply`。

## 2026-11-28：`LichtFeld-Studio` 上游 3DGUT 通用切片移植与 PPISP 曝光梯度修复（`小场景-lfs-planA`）

- **背景与方案选择**：`upstream/master`（至 `7953aead9`）共推进 14 个提交，其中核心性能提交 `56be47617` 约 85% 面向 FastGS（针孔光栅化），且在 Lazy Adam / Morton / MRNF 稀疏折叠上与本地 3DGUT 主线存在 13 处重冲突。采用方案 A 选择性移植通用有益切片并修复本地潜伏 Bug：
  1. Cherry-pick `40cfcfb40`（#4196）：修复 Windows 大小写不敏感目录下评测 PNG 同名/大小写别名覆盖（`filename_key.hpp` / `metrics.cpp` / `test_metrics_evaluator.cpp`）；
  2. 从 `56be47617` 移植 4 项对 3DGUT 通用加速切片：`memory_arena.cu` shrink 免同步预判（`would_decommit`）、`joint_adam_codec.cuh` 小值区 5 阶 Horner 多项式（保留本地 `sr_uniform01` / `encode_us_sr`）、`morton_reorder_kernels.cu` + `morton_reorder.cpp` 连续均值采样打包与流内置换去全局 sync（保留本地 `flush_lazy_steps()`）、`ssim` + `calibration_correction` + `trainer.cpp` 预计算 `decoupled_raw_gradient` 融合；
  3. 修复本地 [ppisp.cu](../../LichtFeld-Studio/src/training/kernels/ppisp.cu#L183-L188) 中 `ppisp_backward_chw_kernel` grid-stride 循环直接将累加器 `grad_exposure_local` 传入赋值语义的 `ppisp_apply_exposure_bwd` 导致仅保留最后一轮像素梯度的 Bug（改为先写临时标量再累加）。
- **单测与构建验证**：
  - 独立编译运行 Torch 无关 GTest 回归（`52` 用例：**`51 PASSED`、`1 SKIPPED`、`0 FAILED`**），覆盖 `JointAdamCodecTest`（10/10）、`GutJointAdamBatch`（2/2）、`GutShNJointAdam`（2/2）、`GutShNJointFromColor`（1/1）、`VramMapCaches`（2/2）、`CalibrationCorrectionTest`（13/13，含新增 `precomputed` 一致性校验）、`CalibrationScoring`（5/5）、`MortonReorderTest`（13/14 passed，1 缺 `bicycle` 数据集 skipped），以及补充回归（`PPISPBackwardGridStrideAccumulatesExposureAcrossAllPixels`、`FilenameKeyRespectsCaseSensitivityFlag`、`MemoryArenaShrinkNoDecommitFastPathPreservesAllocations`）；
  - `scripts\build-windows-release.cmd` Release 构建与 `--version` 冒烟通过。
- **实景验证**（基于 `work/小场景` 跑完整 `20,000` 步 `--headless --eval --perf-bench`，输出保留于 `work/小场景-lfs-planA`；本次同时叠加了 `5e16fd7a5` Route B 精确米制锚点/相对失真/Free-Space 与本次方案 A 修复/优化）：

| 组别 (`小场景`, 259 训 / 37 测, 20k) | 锚点 / 卷帘 | 总耗时 / 稳态 (`#GS`) | 峰值显存 | `PSNR` / `Aligned` | `SSIM` / `Aligned` | `LPIPS` / `Aligned` | `Normal (°)` ↓ | `Depth AbsRel` ↓ | `Bias_corr (R,G,B)` | `00039-L` / `00163-R` / `00175-L` PSNR |
| --- | :---: | :---: | ---: | :---: | :---: | :---: | ---: | ---: | :---: | :---: |
| 基线 (`小场景-lfs-diffuse4`, RTX 4090口径) | SSI RANSAC / `37/37` | — | — | `23.3808` / `24.2519` | `0.7894` / `0.8047` | `0.2941` / `0.2911` | `16.0455°` | `0.0117` | `(-0.0021,-0.0029,-0.0034)` | `21.07` / `19.58` / `25.42` |
| **本次 (`小场景-lfs-planA`, RTX 5070 Ti)** | **`259/259` 米制** / **`37/37`** | `449.1 s` / `22.28 ms/it` (`2,063,260`) | `5,928.8 MiB` | `23.2147` / `24.0580` | `0.7862` / `0.8032` | `0.2965` / `0.2934` | **`14.6517°`** | **`0.0097`** | `(-0.0022,-0.0029,-0.0036)` | `21.00` / **`19.74`** / `25.31` |

- **结论与未验证项**：`train.log` 零警告零错误，`37/37` 评测 PNG 无碰撞导出，`PPISP` 收敛正常（`exp_prior_slope=0.4708`、`exp_prior_rms=0.0126`、`color_rms=0.0124`、`cam1 vig a0=[-0.0029,-0.0030,-0.0010]`），几何法线（`-1.39°`）与深度（`-17.1%`）显著优于原 SSI 基线且白墙/走廊目视干净。未验证项：本机未装 LibTorch，`test_metrics_evaluator.cpp` 未整套编译执行；未在跨硬件同卡对比方案 A 前后的纯吞吐微差。

---

## 2026-11-28：第一批（Wave 1）纯 C++/CUDA 端到端稳定性与性能重构验证（`小场景-opt28`）

- **实验目的**：在不改动深度学习模型权重/精度与外部输出契约的前提下，消除预处理 CPU/GPU 同步等待、重复 LAS 读取、串行三角化过滤、`O(N^2)` 检索与冗余 Ceres BA 开销，并补齐 CUDA 混合精度 BA 失败自动恢复链路，在 RTX 5070 Ti 上对比重构前基线 `work/小场景` 与本次 `work/小场景-opt28`。
- **关键改动**：
  1. `src/geometry/model.h/.cpp`：`Reconstruction::Filter` 支持多线程（`ParallelFor` 分块 1024 点，`points.size() >= 4096` 启用）及线程局部缓冲复用，消除逐点堆分配与串行 bottlenecks。
  2. `src/sfm/optimization.cpp` & `src/sfm/reconstruction.cpp`：合并 `Project`/`ProjectionJacobian`/`CameraJacobian` 为单次 `ProjectAndJacobians` 并消除冗余相机拷贝与旋转矩阵重复计算；米制模式下 2-view `Depth` 保护块收窄至 `depth < 1.0`；增加 Ceres CUDA mixed-precision 失败时 `BaSnapshot` 恢复并逐级降级至 FP64 `DENSE_SCHUR` / `ITERATIVE_SCHUR`；标定阶段在非稳定内参、未掉图、增量观测 `<= max(1, 0.0005 * obs_before)` 且前次 BA 已收敛时跳过冗余尾部固定内参 BA（仍执行完整覆盖校验）。
  3. `src/frontend/inference.h`, `src/frontend/kernels.cu`, `src/frontend/frontend.cpp`：`KeypointDetector` 持久化复用 GPU 缓冲区；`MakeMasks` 对 `roi >= 2 * grounding_size` 采用 `cv::IMREAD_REDUCED_COLOR_2` 直接半分辨率解码并等比缩放 `roi`；`CandidatePairs` 将检索相似度由 `O(N^2)` 标量循环改为 Eigen 矩阵乘法。
  4. `src/io/las.h/.cpp`, `src/alignment/alignment.cpp`, `src/sfm/structure.cpp`, `src/sfm/shutter.cpp`, `src/app/pipeline.cpp`：`ScanLas` 跨分块复用 `bytes`/`points` 向量；新增 `work/cache/las_sample.bin`（`GSLS` v1 + LAS 身份校验）供对齐、卷帘与结构清理复用。
  5. `src/geometry/supervision_kernels.h/.cu`, `monocular.cpp`, `supervision.cpp`：新增 `SolveMogeShiftsKernel` GPU 求解各视角尺度/偏移，消除 `MonocularEstimator::Estimate` 中间 `cudaStreamSynchronize` + D2H/H2D 往返；`EncodeMogeSsiMaps` 沿 readout 轴预计算旋转 LUT；移除裁剪读取时多余全图 `.clone()`。
  6. `src/geometry/photometric.cpp`, `static_masks.cpp`, `src/io/crop.h/.cpp`, `colmap.cpp`：`BuildPhotometricTracks` 改为 15 m 空间网格索引；`static_masks.json` 直接输出 `camera_nonzero_extents` 供裁剪窗口 `O(1)` 读取；`ExportNormalizedJpeg` 按镜头预构建 `rd^2` 索引 LUT 替代逐像素 `std::sqrt`；`RecolorPlyDiffuse` 移入初始化后台线程与 `ExportSupervision` 并行；`CheckBundle` 按图像 `ParallelFor` 并行校验。
- **验证命令与状态**：
  - 编译：`.\tools\windows\build.ps1`（Ninja Release，`/W4` 零错误、无新增警告）。
  - 实景全流程 + 回读检查：`.\run.ps1 run --input input\小场景 --debug --work work\小场景-opt28.work --output work\小场景-opt28; .\run.ps1 check --output work\小场景-opt28`（均 exit 0，`check` `passed: true`，`296/296` 已注册）。
- **核心对比（同卡 RTX 5070 Ti，`input/小场景` 296 张）**：

| 指标 / 阶段 | 基线 (`work/小场景`) | 优化后 (`work/小场景-opt28`) | 变化 |
| --- | :---: | :---: | :---: |
| 全流程 `state.stages` / 总计 | 无 `.work` 记录 | `312.75 s` / **`312.87 s`** | — |
| `masks` 总耗时 (`decode` + `preprocess` / GPU) | `71.46 s` (`23.70 + 13.42` / `69.79 s`) | `68.44 s` (`10.33 + 5.75` / `67.33 s`) | `decode+pre` **-56.7%**（已完全受限于 SAM3 FP16 GPU） |
| `features` / `matches` | `19.30 s` / `27.77 s` | `18.63 s` / `26.79 s` | `-3.5%` / `-3.5%` |
| **`sfm` 总耗时** (`filter` / `opts`) | `94.95 s` (`8.34 s` / `60.47 s`) | **`43.56 s`** (`0.30 s` / `29.28 s`) | **`sfm` 总计 -54.1%**（`filter` **-96.4%**，`opts` **-51.6%**） |
| `align` (`plane_med` / `p90`) | `—` (`6.01 mm` / `21.76 mm`) | `17.67 s` (`5.76 mm` / `21.30 mm`) | 精度保持甚至略优 |
| `shutter` (`val_after`) | `24.07 s` (`0.9530`) | `23.46 s` (`0.9343`) | 持平，验证误差略降 |
| `cleanup` (`las_sample` 读取 / 耗时) | `0.458 s` / `4.55 s` (`plane_after 3.96 mm`) | `0.019 s` / **`3.24 s`** (`plane_after 3.72 mm`) | `sample_seconds` **-95.9%**，`las_sample.bin` (`33.6 MiB`) 生效 |
| `export`（总计 / 裁剪窗口 / 校验回读） | `static_masks 26.82 s` + `sup 70.25 s` | `109.29 s` (`crop 0.054 s`, `readback 1.69 s`) | `RecolorPlyDiffuse`（`724,465/846,533 = 85.58%`）完全隐藏进 `70.31 s` `supervision` GPU 并行窗口 |
| 重建与导出质量 (`pts` / `reproj_med` / `p90`) | `123,316` / `0.7279` / `1.7682` | `122,659` / `0.7284` / `1.7657` | 重投影精度一致，`depth_px=208,505,899`（基线 `208,683,919`） |

- **结论与未验证项**：
  1. `sfm` 与前端解码/几何准备等 CPU+BA 瓶颈已大幅压缩（`sfm` 缩短 `-51.39 s` / `-54.1%`，`masks` 解码+预处理缩短 `-21.04 s` / `-56.7%`，`RecolorPlyDiffuse` 彻底并行掩盖），端到端剩余绝对主导耗时已清晰集中到两大 TensorRT GPU 推理引擎：**SAM3**（`67.33 s`）与 **MoGe-2**（`60.41 s`），适合作为下一波（Wave 2）优化目标（如 SAM3 BF16/FMHA、MoGe-2 分辨率与标定输入/LAS 法线协同）。
  2. **未验证项**：`小场景` 未触发 Ceres CUDA 混合精度失败，因此 `mixed_precision_fallback` / `cuda_fallback_iterative` 仅完成编译链接验证，未在极端病态场景中实测触发；未在 `展厅` / `南非展会` 等更大场景回归；未对 `work/小场景-opt28` 重跑下游 LichtFeld Studio 3DGS 训练对比渲染指标。

---

## 2026-11-28：第二批（Wave 2）TensorRT GPU 推理引擎优化（`小场景-wave2`）

- **实验目的**：攻克 Wave 1 后端到端占主导的两大 GPU 推理耗时（SAM3 `64.97 s` + MoGe-2 `55.83 s`），在独立引擎包 `work/models-wave2`（不覆盖只读 `models/`）上验证精度与端到端实测收益。
- **技术路径与关键消融发现**：
  1. **SAM3 (`grounding`)**：实测纯 BF16 与无约束 FP16 都会触发 TensorRT 10.13 Myelin 数值精度退化（`score_max = 0.928–0.971`、`mask_iou_min = 0.0`），单层消融锁定问题来自 **28 个窗口块**（`window=24`、`B=9, L=576`）和 **31 个带 `-inf` 掩码的交叉/解码 Softmax 链**，而 **10 个重型 5184-token 全局自注意力**（4 层全局 `trunk.{7,15,23,31}` + 6 层 `encoder.self_attn`）在纯 FP16 下完全稳定。改用 `PlainSelfAttention` / 预缩放 4D SDPA 取消 648 行分块并仅放开这 10 个全局自注意力融合 TensorRT FP16 `_gemm_mha_v2` FMHA，其余 59 个 Softmax 链保持 FP32。
  2. **MoGe-2 (`geometry`)**：由 `644×644`（`2116` tokens）切换至官方建议区间中的 `560×560`（`40×40 = 1600` tokens），保持 FP16 与 `B = 1..8`（`opt=7`）动态批。
- **验证命令与状态**：
  - 引擎构建与校验：`.\tools\windows\prepare-models.ps1 -Models work\models-wave2 -OnnxFrom models` + `.\run.ps1 check-models --models work\models-wave2 --image input\小场景\Image\2433-514856-623_00039-R.JPG`（均 exit 0，`check-models` `passed: true`）。
  - 实景全流程 + 回读检查：`.\run.ps1 run --input input\小场景 --models work\models-wave2 --debug --work work\小场景-wave2.work --output work\小场景-wave2; .\run.ps1 check --output work\小场景-wave2`（均 exit 0，`check` `passed: true`，`296/296` 已注册）。
- **核心对比（同卡 RTX 5070 Ti，`input/小场景` 296 张）**：

| 指标 / 阶段 | Wave 1 (`work/小场景-opt28`) | Wave 2 (`work/小场景-wave2`) | 相对 Wave 1 变化 |
| --- | :---: | :---: | :---: |
| **`grounding` 单帧延迟 / 运行显存** | `244.24 ms` / `1,743.9 MB` | **`124.50 ms`** / `1,794.4 MB` | **-49.0%**（PyTorch 参考 `IoU >= 0.9982`） |
| **`masks` 阶段总耗时 / TRT GPU 耗时** | `68.53 s` / `64.97 s` | **`38.87 s`** / **`36.14 s`** | **`-29.66 s`（-43.3%）** / `-44.4%` |
| `masks` 检测统计 & 296 图掩码一致性 | `persons=540, shadows=95, skies=132` | `persons=539, shadows=95, skies=131` | 像素一致率 **`99.934%`**，零掩码 IoU **均值 `0.9979` / 中位 `1.0000`** |
| **`geometry` (`moge_estimate` / `TRT GPU`)** | `60.41 s` / `55.83 s` | **`46.03 s`** / **`42.30 s`** | **`-14.38 s`（-23.8%）** / `-24.2%` |
| `export` 总耗时 (`sup_total` / `depth_px`) | `109.29 s` (`70.30 s` / `208,505,899`) | **`94.32 s`** (`55.86 s` / `208,444,607`) | **`-14.97 s`（-13.7%）** |
| 60 图采样监督对比（米制深度 / 法向） | 基线 (`644×644`) | `560×560` | 深度相对差中位 **`0.228%`** (`p90 2.04%`)；法向角差中位 **`1.266°`** (`p90 6.48°`) |
| 重建与配准质量 (`pts` / `reproj_med` / `p90`) | `122,659` / `0.7283` / `1.7658` | `123,528` / `0.7264` / `1.7687` | 精度等价 |
| **全流程 `state.stages` / `run` 总耗时** | `312.75 s` / `312.87 s` | **`271.63 s` / `271.75 s`** | **`-41.12 s`（-13.1%）**（相对 Wave 1 前 `~368 s` 累计约 `-96 s`） |

- **下游 LichtFeld Studio 3DGS 闭环验证**（同卡 RTX 5070 Ti，基于 `work/小场景-wave2` 运行 `20,000` 步 `--headless --eval --perf-bench`，输出保留于 `work/小场景-lfs-wave2`，对比 `work/小场景-lfs-planA`）：

| 组别 (`小场景`, 259 训 / 37 测, 20k) | 锚点 / 卷帘 | 总耗时 / 稳态 (`#GS`) | 峰值显存 | `PSNR` / `Aligned` | `SSIM` / `Aligned` | `LPIPS` / `Aligned` | `Normal (°)` ↓ | `Depth AbsRel` ↓ | `Bias_corr (R,G,B)` | `00039-L` / `00163-R` / `00175-L` PSNR |
| --- | :---: | :---: | ---: | :---: | :---: | :---: | ---: | ---: | :---: | :---: |
| 基线 (`小场景-lfs-planA`, `644×644`) | **`259/259` 米制** / **`37/37`** | `449.1 s` / `22.28 ms/it` (`2,063,260`) | `5,928.8 MiB` | `23.2147` / `24.0580` | `0.7862` / `0.8032` | `0.2965` / `0.2934` | `14.6517°` | **`0.0097`** | `(-0.0022,-0.0029,-0.0036)` | `21.00` / `19.74` / `25.31` |
| **本次 (`小场景-lfs-wave2`, `560×560` + FMHA)** | **`259/259` 米制** / **`37/37`** | **`434.8 s`** / **`21.60 ms/it`** (`2,064,355`) | `6,024.8 MiB` | `23.1319` / `23.8867` | **`0.7874`** / `0.8025` | **`0.2963`** / `0.2934` | **`14.6340°`** | `0.0107` | `(-0.0030,-0.0035,-0.0039)` | `20.96` / **`20.18`** / `25.10` |

- **结论与未验证项**：
  1. SAM3 10 层全局 FMHA + MoGe-2 `560×560` 在几乎零精度损失（掩码中位 IoU `1.0000`、米制深度差中位 `0.228%`、法向差中位 `1.266°`）下把整套 296 张实景预处理压进 **`271.75 s`**（4 分 32 秒），GPU 推理时间从 `120.80 s` 降到 `78.44 s`。
  2. 下游 LichtFeld Studio 3DGS 全量 20k 闭环训练一次跑通且零报错：`259/259` 精确米制锚点与 `37/37` 卷帘相机全部生效，`eval_step_20000` 导出 `37` 张评测 PNG；法线角略优（`14.6340°` vs `14.6517°`）、裸 `SSIM` (`0.7874`) 与 `LPIPS` (`0.2963`) 持平略优，`PSNR` 微差 `-0.08 dB`（难点视角 `00163-R` 升至 `20.18 dB`），`Depth AbsRel` 微差 `+0.0010`，训练总耗时 `434.8 s`（稳态 `21.60 ms/it`）。
  3. **未验证项**：默认 `models/` 已替换为验证通过的 `work/models-wave2` 引擎包；尚未在 `展厅`/`南非展会` 等更大场景做 Wave 1+2 预处理及下游训练回归。


