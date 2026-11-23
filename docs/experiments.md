# 实验记录

统一记录 GeosunPre 关键算法、配置和产物规格改动后的实测对照数据与取舍结论，便于跨对话延续上下文；只写已实际观测到的数据。

## 2026-11-22：小场景异步双鱼眼卷帘物理校验、等时长先验与走廊 Rig 约束修复

- 背景与诊断：在 `work/1122/小场景`（296 幅 `ShareUAV CM2000-2.55D` 3600×3600 异步双鱼眼，L/R 约 56 ms 交替触发）及下游 3DGS（`configs/trainer.json` 30k + `--eval-all`）中，地砖等近景表面出现双边/模糊（如 `00071-L`）且走廊窗户及地面局部大幅错位（如 `00072-L`）。定位为两项叠加因素：
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

## 2026-11-22：高置信平面 LAS 法向优先 + MoGe-2 回填混合法向监督（`las_planar_moge2_fill`）

- **背景与核心改动**：此前解耦监督（`normal_source = "moge2_faces"`）全图采用 MoGe-2 单图法向、仅用 LAS 面元做 `> 35°` 幻觉剔除，在弱纹理墙面和地面上仍受单目预测跨视角 `2°–4°` 漂移影响。改为在 [monocular.h](../src/geometry/monocular.h) / [monocular.cpp](../src/geometry/monocular.cpp) 与 [supervision.cpp](../src/geometry/supervision.cpp) 中保持原有 `las_moge_cleaned` 深度清洗与 `> 35°` 冲突门控不变，在非边缘且通过 RANSAC 清洗的实体表面上优先写入多视角几何一致的平面 LAS 面元法向（`las_normal_used_pixels`），仅在 LAS 缺测、MoGe 边缘带及非平面细节处由 MoGe-2 法向回填（`moge_normal_fill_pixels`）；`supervision.json` 保持 `source = "las_cleaned_moge2_normals"`、`depth_source = "las_moge_cleaned"`，更新 `normal_source = "las_planar_moge2_fill"`，与下游 `LichtFeld-Studio` `las_prior_sidecar.cpp` 完全兼容。

### 实景对比：`小场景`（296 幅）与 `展厅`（588 幅）

两套实景均全流程跑通并经 `run.ps1 check` 校验通过（`passed: true`，`296 / 296` 与 `588 / 588` 注册），输出目录（`work/小场景-las-norm-out` 与 `work/展厅-las-norm-out`）保留在本地供下游 3DGS 训练对照。

| 指标 | `小场景` 纯 MoGe (`小场景-imu-sfm-out`) | `小场景` LAS 优先 (`小场景-las-norm-out`) | `展厅` 纯 MoGe (`展厅-imu-sfm-out`) | `展厅` LAS 优先 (`展厅-las-norm-out`) |
| --- | :---: | :---: | :---: | :---: |
| `normal_source` | `moge2_faces` | `las_planar_moge2_fill` | `moge2_faces` | `las_planar_moge2_fill` |
| 有效深度像素 `depth_pixels` | `171,255,710` | `171,302,367`（持平 `+0.03%`） | `309,680,050` | `309,724,759`（持平 `+0.01%`） |
| 有效法向像素 `normal_pixels` | `203,330,932` | `203,939,965`（`+0.30%`） | `371,896,047` | `373,320,888`（`+0.38%`） |
| LAS 平面法向像素 (`las_normal_used`) | `0` | **`155,329,822`（`76.2%`）** | `0` | **`264,669,380`（`70.9%`）** |
| MoGe-2 回填像素 (`moge_normal_fill`) | `203,330,932` | `48,610,143`（`23.8%`） | `371,896,047` | `108,651,508`（`29.1%`） |
| `normal_disagree_dropped` | `5,431,670` | `5,401,664` | `22,524,721` | `22,520,963` |
| `penetration_normal_dropped` | `447,144` | `441,808` | `1,822,754` | `1,787,728` |
| 抽样有效区法向变化（`>0.6°` 占比 / 角差 med / p90） | — | `84.0%` / `3.26°` / `9.98°` | — | `70.8%` / `3.86°` / `14.55°` |

- **跨视角几何一致性评测（`小场景`，逐行卷帘重投影 + `<= 2 px` 双向回环去遮挡）**：
  - **同镜头（相邻 1–2 帧，~455 万像素）**：跨视角深度中位/P90 差 `1.04 / 6.17 mm -> 1.05 / 6.21 mm`（`< 2 cm` 占比保持 `98.4%`）；世界系法线夹角由 `2.69° / 7.77°` 降至 **`0.28° / 1.06°`**（`< 5°` 占比 `77.8% -> 97.1%`），点-面偏移由 `0.33 / 2.04 mm` 收紧至 **`0.29 / 1.80 mm`**。
  - **跨镜头 L→R（近同时刻，~17.6 万像素）**：深度中位/P90 差 `1.07 / 4.75 mm -> 1.09 / 4.90 mm`（`< 2 cm` `99.3% -> 99.2%`）；世界系法线夹角由 `2.80° / 7.52°` 降至 **`0.27° / 0.74°`**（`< 5°` 占比 `77.6% -> 98.3%`），点-面偏移由 `0.26 / 1.19 mm` 收紧至 **`0.24 / 1.01 mm`**。
- **结论与待验证**：两场景有效监督法向中约 `71%–76%` 的墙面、地面等高置信度平面像素由多视角绝对一致的毫米级 LAS 面元法向接管，在被替换区域纠偏了原 MoGe-2 单目预测 `3.3°–3.9°`（P90 `10.0°–14.5°`）的视角漂移并将 `小场景` 跨视角法线一致性从 `~2.7°–2.8°` 压低至 `~0.27°–0.28°`；`depth_pixels` 完整保持，有效 `normal_pixels` 略有提升且零新增 GPU 开销。待在下游 `LichtFeld-Studio` 用 `work/小场景-las-norm-out` 与 `work/展厅-las-norm-out` 启动 3DGS 试训评估 novel view 法线与渲染改善。

## 2026-11-23：3DGUT 沿光线法向/颜色一致性正则（LichtFeld-Studio）与 `configs/trainer.json` 升级

- **问题与核心改动**：在 `work/小场景-las-norm-out` 高处俯视 novel view 下，旧版 `configs/trainer.json`（含局部双边网格 + `sh_degree = 0` + 全局不透明度衰减）及仅改单像素正则的 `exp3/exp4` 仍在 1F 白墙上从 `3k → 30k` 渐进长出灰白糊斑与贴墙倾角瓦片（根因在于 pixel-wise 混合平均法线与 RGB 对沿光线正负倾斜短轴及互补黑白微粒存在抵消盲区）。我们在下游 `LichtFeld-Studio` 中实现两项**无量纲、与场景尺度无关且默认 `0.0`** 的 GUT 沿光线一致性正则并完成消融升级：
  1. **沿光线法向方差正则 `normal_variance_weight`**（`--normal-variance-weight`，新增 `src/training/kernels/normal_variance_loss.hpp` / `src/training/kernels/normal_variance_loss.cu`，经 `grad_normal` / `grad_alpha` 反传并尊重 `normal_start_fraction`、ROI 与 `SegmentAndIgnore` 掩码权重）：直接惩罚每条光线上各高斯单位短轴相对混合平均法线的离散度 $\operatorname{var}_n = 1 - \|N\|^2 / \alpha^2$。
  2. **GUT 沿光线颜色方差正则 `color_variance_weight`**（`src/training/rasterization/gsplat/RasterizeToPixelsFromWorld3DGSBwd.cu` / `src/training/rasterization/gsplat_rasterizer.cpp` / `src/core/parameters.cpp`）：在 GUT 光栅反向阶段启用 SteadySplats Eq. 7 沿光线 RGB 方差惩罚，并放行 `gut = true` 下与掩码模式、ROI 权重协同使用。
  3. **舍弃实验性世界尺度上限 `max_scale_reg` / `max_scale_threshold`**：`exp5` 实测 `0.1 m` 硬帽会重伤 4F 高层玻璃电梯与稀疏大面（全场景 PSNR `-0.95 dB`、部分视角跌 `-3.38 dB`），且带物理米制量纲无法通用，已在清理提交时彻底移除。

### 1. `小场景-las-norm-out` 核心消融对照（296 视角，30k `--eval-all`）

完整 8 组消融产物、逐视角评测与对比渲染保留在 `work/las-norm-ablation/`。

| 实验组 | 关键设置（相对旧配置） | 耗时 | `#GS` | Mask PSNR / SSIM / LPIPS | Aligned PSNR / SSIM / LPIPS | Bias (raw) | 法线角 (°) ↓ | 深度 med / P90 (mm) | `alpha < 0.98` ↓ | `1F_L`: `od / >15cm / dark / ang` |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---|
| `exp0-baseline` | 旧 `trainer.json`（`sh=0`, 双边网格） | `843.8s` | `2,984,778` | `24.2966 / 0.8293 / 0.2415` | `25.4524 / 0.8537 / 0.2374` | `(+0.042,+0.041,+0.047)` | `9.0262` | `6.33 / 34.33` | `0.3147` | `5.38x / — / 10.8% / —` |
| `exp3-reg-sh1` | PPISP + `dc=0.005, erank=0.008, sh=1, nflat=0.01, stop_refine=4500` | `873.6s` | `2,994,286` | **`24.7914 / 0.8343 / 0.2399`** | **`26.2151 / 0.8597 / 0.2353`** | `(+0.004,+0.001,+0.005)` | `8.2109` | **`6.15`** / `33.63` | `0.2681` | `6.11x / 477 / 8.8% / 50.3°` |
| `exp5-gut-ray-consist` | + `nflat=0.03, nvar=0.15, cv=0.5@3k, max_s=0.002@0.1m` | `851.7s` | `2,997,379` | `23.8446 / 0.8175 / 0.2547` | `25.0517 / 0.8431 / 0.2500` | `(-0.009,-0.009,-0.008)` | `8.0658` | `7.01 / 38.25` | `0.3024` | `4.66x / 0 / 2.4% / 15.4°` |
| `exp6-nvar-only` | + `nflat=0.03, nvar=0.15`（纯法向方差） | `858.4s` | `2,993,935` | `24.5679 / 0.8277 / 0.2471` | `25.9365 / 0.8527 / 0.2426` | `(+0.004,+0.004,+0.007)` | **`7.0754`** | `6.26 / 33.31` | **`0.2564`** | `6.82x / 559 / 6.2% / 19.1°` |
| **`exp7-nvar-cv-balanced`** | + **`nflat=0.02, nvar=0.06, cv=0.18@5k`** | `877.3s` | `2,994,590` | `24.4838 / 0.8286 / 0.2452` | `25.8417 / 0.8542 / 0.2405` | **`(-0.001,-0.001,+0.001)`** | `7.6668` | `6.17` / **`33.12`** | `0.2653` | `8.52x / 634 / 4.3% / 29.8°` |
| `exp8-nvar-cv-polish` | + `nflat=0.02, nvar=0.10, cv=0.30@5k` | `870.1s` | `2,994,451` | `24.3410 / 0.8255 / 0.2484` | `25.6741 / 0.8511 / 0.2437` | `(-0.002,-0.003,-0.000)` | `7.6408` | `6.29 / 33.36` | `0.2671` | `9.01x / 653 / 3.2% / 23.0°` |

### 2. 跨场景回归验证：`展厅-las-norm-out`（588 视角，1,208 m²，30k，清理后最终二进制）

使用移除 `max_scale` 后的最终 Release 编译版对 `work/展厅-las-norm-out` 进行完整 30k A/B 验证（结果保留于 `work/las-norm-ablation/zhanting-{baseline,exp7}`）：

| 指标 | `zhanting-baseline`（旧配置） | `zhanting-exp7`（新 `trainer.json`） | 变化 |
|---|---:|---:|---|
| 耗时 / `#GS` | `950.4s` / `2,999,903` | **`899.1s`** / `2,988,447` | **`-5.4%`** |
| `Mask PSNR / SSIM / LPIPS` | **`26.1824` / `0.8511` / `0.2657`** | `26.1516` / `0.8486` / `0.2717` | PSNR `-0.031 dB` |
| `Aligned PSNR / SSIM / LPIPS` | `26.8482` / `0.8651` / `0.2630` | **`26.8578`** / `0.8632` / `0.2685` | **Aligned PSNR `+0.010 dB`**（max 移 `5.49 -> 4.15 px`） |
| `Normal angle (deg)` / `Depth AbsRel` / 中位 (mm) ↓ | `9.9503°` / `0.0060` / `8.7783` | **`7.6643°` / `0.0059` / `8.1261`** | **法线 `-2.29°` / 深度 `-0.65 mm`** |
| `alpha < 0.98` 空洞占比 ↓ | `0.2284` | **`0.1854`** | **`-4.30 pp`** |
| `depth_prior_err_p90_mm` ↓ | **`52.5313`** | `73.0369` | 见下方排查确认 |
| 全局 `OOG` / 细针 `>5` / `>10` ↓ | `30.30%` / `21.55%` / `5.51%` | **`0.04%` / `3.02%` / `0.92%`** | **退化高光/超色域近清零，细针 `-86%`** |

- **`p90` 统计差异诊断与三维验证**：在 `882,665` 原始 LAS 点多尺度 3D 体素偏差审计中，`zhanting-exp7` 在有效不透明层全面优于基线（`opac > 0.5`: `off > 5 cm` `2.50% vs 2.63%`、`off > 15 cm` `0.14% vs 0.23%`、`2,483 vs 2,702` 颗；`opac > 0.05`: `off > 30 cm` `709 vs 823`、`s_max` 中位 `9.16 vs 22.93 cm`；`>25 cm` 且 `opac > 0.1` 远端浮渣颗粒由 `1,475` 颗降至 **`978` 颗（`-33.7%`）**、离群空间体素格由 `876` 格降至 **`680` 格**）；`--test-every 50`（12 视角）上 `exp7` 的 `p90` 同样优于基线（**`36.35 vs 36.90 mm`**），全量 `588` 视角 `p90` 上升系个别低先验覆盖贴桌近距视角、透明展柜玻璃反光与 `alpha < 0.98` 压低 `-4.30 pp` 后更多边缘难像素越过 `kDepthPriorMinAlpha = 0.5` 进入尾部统计所致，视觉抽查 `eval_step_30000` 图像确认无几何退化。

### 3. 结论、配置升级与未验证边界
- **结论与主线升级**：选取 `exp7` 作为新 Pareto 甜点更新至 [configs/trainer.json](../configs/trainer.json)（PPISP 替代局部双边网格、`sh_degree = 1 / sh_rest_reg = 0.001`、`dc_reg = 0.005`、`erank_reg = 0.008`、`scale_reg = 0.005`、`gradient_loss_weight = 0.0`、`start_refine = 1200 / stop_refine = 4500`、`opacity_decay = 0.0 / scale_decay = 0.0`、`normal_loss_weight = 0.1`、`normal_prior_depth_weight = 0.05`、`normal_prior_decay_start = 0.95`、`normal_consistency_weight = 0.1`、`normal_flatten_weight = 0.02`、`normal_variance_weight = 0.06`、`color_variance_weight = 0.18`、`color_variance_warmup_iters = 5000`，保留原 `save_steps: [7000, 30000]`）；极大面积纯白墙极端场景可酌情手动调高至 `exp8` 挡（`0.10 / 0.30`）。
- **验证与未验证项**：`LichtFeld-Studio` Release 全量编译（`171/171`）、CLI `--help` 与 200 步联合冒烟测试以及双实景全量 30k 对比均已完成；`tests/test_color_variance.cpp` 已同步补充 GUT 掩码放行与 `NormalVarianceParameters` 用例，但 `build-windows-release` 开启 `BUILD_TESTS=OFF`，测试可执行文件尚未单独编跑。

## 2026-11-23：预处理导出阶段自适应生成 `trainer.json`（LFS 零改动）

- **问题与设计**：此前下游固定共用同一份 [configs/trainer.json](../configs/trainer.json)（`iterations = 30000`、`max_cap = 3000000`），未随图像规模（数百至数千张）、ROI 几何体素复杂度及训练机显存自适应调整。在**完全不修改 `LichtFeld-Studio`** 的前提下，在 `GeosunPre` 侧完成全链路自适应配置生成：
  1. [CMakeLists.txt](../CMakeLists.txt) 将 `configs/trainer.json` 内嵌编译为 `DefaultTrainerConfig`；新增 [trainer_config.h](../src/io/trainer_config.h) / [trainer_config.cpp](../src/io/trainer_config.cpp)，由 `ExportBundle` 在导出期自动生成 `<output>/trainer.json`（同步写入 `manifest.json` 的 `trainer_config` 摘要并在 `check` 中校验；存量交付包可用 `geosunpre trainer-config --output DIR [--vram-gb GB] [--trainer-config JSON]` 补生成）。
  2. **迭代步数与调度同步缩放**：按 `N_train`（已注册图像数扣除 `test_every` 留出比例）以 `550` 张为锚点幂律推导 `iterations = round_1000(clamp(base_iters * (N_train / 550)^0.6, 20000, 120000))`，并按同一比例缩放 `start_refine`、`stop_refine`、`reset_every`、`refine_every`、`morton_reorder_interval`、`sh_degree_interval`、`grow_until_iter`、`color_variance_warmup_iters`、`eval_steps` 与 `save_steps`，保持 `steps_scaler = 1.0` 避免重复缩放。
  3. **高斯上限与初始化点云协同**：取几何需求 `cap_geom = 3.0 * geom_points`（优先取 `roi_voxel_points_before_cap`）与视角支撑 `cap_views = 8000 * N_train` 较小值，受显存硬上限 `cap_vram = round_100k(clamp((vram_gb - 2.5) * 1800000, 4000000, 24000000))`（`8 GB -> 9.9M`、`24 GB -> 24M`）钳位并下限保护 `1.5M`，写出 `max_cap`（LFS 自动按 `grow_fraction = -1` / `shs_lr = -1` 联动）；`ExportLasPly`（[las.cpp](../src/io/las.cpp)）同步记录 `roi_voxel_points_before_cap` 并将 PLY 封顶联动为 `min(2500000, max(100000, round(0.75 * max_cap)))`。
  4. [bundle/main.cpp](../bundle/main.cpp) 调整为训练优先加载 `<delivery>/trainer.json`，不存在时回退包内默认模板；[output.md](output.md) 已同步更新契约。

### 三场景实景验证（`RTX 4090`，`23.99 GiB` 显存，`--eval-all`，无重复重跑基线）

- `小场景` 对已有交付目录 `work/小场景` 通过 `geosunpre trainer-config` 生成 `trainer.json` 并通过 `check`（另验证 `--vram-gb 8` 时 `cap_vram = 9,900,000`）；`展厅` 与 `篮球场` 通过 `run.ps1 run` 端到端预处理分别导出至 `work/展厅`（`709.6 s`，`588/588` 注册）与 `work/篮球场`（`765.7 s`，`536/536` 注册）。三组训练产物均保留在 `work/adaptive-test/`。

| 场景 / 方案 | `N_reg` / `N_train` | `geom_points` (`roi_voxel`) | `iterations` / `max_cap` / `ply_cap` | 训练耗时 | 裁切后 `#GS` | 采样显存 (`nvidia-smi`) | `Mask PSNR / SSIM / LPIPS` | `Aligned PSNR` | 法线角 (°) ↓ | 深度 med / p90 (mm) ↓ |
| --- | :---: | ---: | :---: | ---: | ---: | ---: | --- | ---: | ---: | --- |
| `小场景` 基线（`exp7` `30k/3M`） | `296` / `296` | `~845k` | `30,000 / 3,000,000 / 2,500,000` | `877.3 s` | `2,994,590` | — | **`24.4838 / 0.8286 / 0.2452`** | **`25.8417`** | **`7.6668°`** | `6.17 / 33.12` |
| `小场景` 自适应（`work/adaptive-test/小场景/metrics_report.txt`） | `296` / `259` | `845,322` | `20,000 / 2,100,000 / 1,575,000` | **`492.8 s`**（`-43.8%`） | `2,096,943` | `6,068 MiB`（`11,400` 步采样） | `23.8773 / 0.8122 / 0.2734` | `24.9136` | `7.9020°` | **`5.9110 / 30.0103`** |
| `展厅` 基线（`zhanting-exp7`） | `588` / `588` | `~882k` | `30,000 / 3,000,000 / 2,500,000` | `899.1 s` | `2,988,447` | — | **`26.1516 / 0.8486 / 0.2717`** | **`26.8578`** | `7.6643°` | **`8.1261`** / `73.0369` |
| `展厅` 自适应（`work/adaptive-test/展厅/metrics_report.txt`） | `588` / `514` | `881,016` | `29,000 / 2,600,000 / 1,950,000` | **`863.9 s`**（`-3.9%`） | `2,585,606` | `9,084 MiB`（`17,300` 步采样） | `26.1391 / 0.8474 / 0.2763` | `26.8017` | **`7.6382°`** | `8.3586` / **`64.7211`** |
| `篮球场` 自适应（`work/adaptive-test/篮球场/metrics_report.txt`） | `536` / `469` | `5,143,902` | `27,000 / 3,800,000 / 2,500,000` | `835.0 s` | `3,758,049` | `8,666 MiB`（`17,600` 步采样） | `19.1562` / `0.5861 / 0.3393` | `19.3709` | `13.5641°` | `9.6396 / 113.1315` |

注：三组自适应评测均开启 `--eval-all`，卷帘相机分别正确激活 `296/296`、`588/588` 与 `536/536`。

### 结论与未验证项
- **已验证生效**：
  - 全链路从 `GeosunPre` 预处理导出、PLY 点云封顶联动、`check` 校验到 `LichtFeld-Studio` 直接读取 `trainer.json` 完整闭环，无二次步数缩放与 OOM；
  - `展厅`（中等规模）自适应将容量收敛至 `2.6M / 29k` 步，耗时降低 `3.9%`、模型体积缩减 `13.5%`，PSNR 仅 `-0.0125 dB` 而法线角与深度 `p90` 均更优；
  - `篮球场`（高几何密度室外大场景）突破 `3.0M` 固定上限抬升至 `3.8M`，实测采样显存约 `8.5 GiB`；
  - `小场景`（`296` 张图）揭示出小样本区间（`< 300` 图）将步数压到 `20k` + 容量压到 `2.1M` 时虽提速 `43.8%` 且深度误差更低（`5.91 / 30.01 mm`），但外观细节付出 `-0.61 dB` PSNR 代价；后续如需保小场景细节，可微调下限（如 `I_min -> 24000~25000` 或微升小场景容量下限）。
- **未验证项**：
  - 未在真实 `8 GB` 物理显卡上实跑训练，仅验证了 `--vram-gb 8` 的参数计算路径（`cap_vram = 9.9M`）；
  - 未在几千张图（`2,000–8,000` 张）的超大实景数据集上做端到端训练耗时与显存压测，也未通过 `Geosun.exe` 打包体做端到端调用回归（仅核查 `bundle/main.cpp` 编译链接与路径优先级逻辑）。


## 2026-11-23：GUT 训练变慢排查与性能修复（`GeosunPre` + `LichtFeld-Studio`）

- **排查定位（基于 `work/小场景` `3000` 步控制变量对比与 `cuobjdump` 分析）**：
  1. **上游 `#4147` 默认启用 `calibration_correction`**：`LichtFeld-Studio` 将 `parameters.hpp` 中 `calibration_correction` 默认改为 `true`；`configs/trainer.json` 缺省该字段导致 `iter >= 1000` 后每步执行 2D 网格扭曲拟合与先验随动（`warp`/`follow`/`accumulate`/`update`，在 `小场景` `3000` 步下耗时从 `46.490 s / 64.5 iter/s` 增至 `49.094 s / 61.1 iter/s`），且开启 `--eval-all` 时每张图额外运行 `align_for_scoring` 4 层金字塔配准（`小场景` `296` 视角单次评测由 `0.2708 s/image` 升至 `0.3178 s/image`、`1000` 步含一次评测总耗时由 `93.158 s` 升至 `107.614 s`）。由于 `GeosunPre` 已提供精确逐行卷帘（`rolling_shutter.json`）与 LAS 先验，训练期与评测期无需再启用该扭曲拟合。
  2. **上游 `#4133` 引起的同一步重复 `fit_depth_prior`**：`GeosunPre` 的先验尺寸（如 `1018×1024` / `1024×1024`）小于 `resize_factor=2` 后的训练图像（`2816×2832` / `2832×2832`），命中 `pipelined_depth_own_size_ == true`，导致 `trainer.cpp` 在 `free_space` 准备与 `depth_loss` 阶段同一步内对 `pipelined_depth_` 调用两次 `fit_depth_prior`。
  3. **GUT 反向核 `bwd_splat_kernel` 的共享内存与全局读取开销**：`RasterizeToPixelsFromWorld3DGSBwd.cu` 无条件为 `s_cv_target` / `s_cv_buf` 分配 `1,280 B` 共享内存（使 `CDIM=8` 达到 `10,240 / 10,496 B`）并保留内层分支，且存在 `pixel_weight` 时每像素重复读取 `*color_variance.pixel_weight_sum`。
  4. **预期内配置变化贡献**（保留不变）：`configs/trainer.json` 从 `sh_degree=0` / 关闭 PPISP 升级到 `sh_degree=1` / `use_ppisp=true` 在 `小场景` `3000` 步下分别对应约 `2.0 s` 与 `1.7–2.0 s` 增量（关闭 PPISP `44.819 s`，再关正则与 SH1 `42.850 s`，再关深度/法线损失 `37.235 s`）；`normal_variance_weight` 与 `color_variance_weight` 本身在无共享内存挤占后运行开销可忽略。

- **改动**：
  - [configs/trainer.json](file:///d:/Dev/GeosunPre/configs/trainer.json)：显式添加 `"calibration_correction": false`。
  - [trainer.cpp](file:///d:/Dev/LichtFeld-Studio/src/training/trainer.cpp)：在 `!fit` 且分辨率匹配时，`depth_loss` 直接复用前序已适配的 `free_space_depth` 并通过 `sync_to_stream` 同步到 `depth_stream`，消除每步第二次 `fit_depth_prior`。
  - [RasterizeToPixelsFromWorld3DGSBwd.cu](file:///d:/Dev/LichtFeld-Studio/src/training/rasterization/gsplat/RasterizeToPixelsFromWorld3DGSBwd.cu)：为 `rasterize_to_pixels_from_world_3dgs_bwd_splat_kernel` 增加 `kColorVariance` 编译期模板参数（关闭时将 `s_cv_target`/`s_cv_buf` 缩至 1 元素，共享内存降回 `8960 / 9216 B`，`LOCAL:0`），并在开启分支前由 warp `lane 0` 读取 `*color_variance.pixel_weight_sum` 后 `__shfl_sync` 广播。

- **验证与未验证项**：
  - `LichtFeld-Studio` Release 与 `GeosunPre` Release（`geosunpre.exe` / `geosun.exe`）均编译通过；`geosunpre trainer-config --vram-gb 24` 验证生成配置含 `"calibration_correction": false`。
  - `work/小场景` 同参数（`max_cap = 2,100,000`、`iterations = 3,000`）基准：`benchA` 修复前默认 `49.094 s (61.1 iter/s)` → 修复后 warmup 期（`warmup=2999`）提速至 `45.170 s (66.4 iter/s)`（`-8.0%`）；全程开启 color variance（`warmup=0`）为 `46.912 s (63.9 iter/s)`；直接使用更新后 `configs/trainer.json`（`start_refine=1200`）`3,000` 步为 `41.330 s (72.6 iter/s)`。
  - **未验证项**：未重跑完整 `20k–30k` 实景训练及 `LichtFeld-Studio` 单元测试套件（当前 Release 构建 `BUILD_TESTS=OFF`）。

## 2026-11-23：恢复 MRNF 后期清理与衰减参数以消除展厅空域漂浮点

- **问题定位**：与 `1.1.4` 相比，`展厅`点云视图中存在大量未被清除的空域漂浮点。排查确认根因为提交 `ecf8fbe` 将 [configs/trainer.json](../configs/trainer.json) 中的 `stop_refine` 提前至 `4500`（`LichtFeld-Studio` MRNF 在 `iter >= stop_refine` 时冻结拓扑并跳过 `refine()` 中的 `soft_delete_rows`）并将 `opacity_decay` / `scale_decay` 归零，导致 `4500` 步之后被 `free_space_loss`、`dc_reg` 及 `color_variance` 压暗/压低不透明度的空域漂浮点无法被 prune。
- **改动**：调整 [configs/trainer.json](../configs/trainer.json) 细化与外观参数，在不修改 `LichtFeld-Studio` 代码的前提下恢复中后期持续清理与衰减：
  - `start_refine`: `1200` → `1500`
  - `stop_refine`: `4500` → `28500`
  - `opacity_decay`: `0.0` → `0.002`
  - `scale_decay`: `0.0` → `0.002`
  - `sh_degree`: `1` → `0`
  - 其余 `ecf8fbe` / `c5a1f16` 正则与外观升级（`use_ppisp=true`、`normal_variance_weight=0.06`、`color_variance_weight=0.18@5000`、`calibration_correction=false`）保持不变。
- **验证与未验证项**：
  - 用户已完成 `小场景` 回归验证，效果与此前相当，确认保留 `normal_variance_weight` 与 `color_variance_weight` 后放宽 `stop_refine` 至 `28500` 未引发白墙退化；
  - `GeosunPre` Release 重编通过，内嵌默认配置及生成 `trainer.json` 已核验包含上述更新参数；
  - **未验证项**：尚未用该配置重跑 `展厅` 完整训练并复查点云视图漂浮点清理结果。
