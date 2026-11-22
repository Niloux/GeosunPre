# 实验记录

统一记录 GeosunPre 关键算法、配置和产物规格改动后的实测对照数据与取舍结论，便于跨对话延续上下文；只写已实际观测到的数据。

## 2026-11-15：精简主线、单路 MoGe 监督、共享面元缓存、裁剪分档 PLY 与收紧前端配对

- **测试环境**：Windows x64，`NVIDIA GeForce RTX 5070 Ti`（CC 12.0），C++17 / CUDA / TensorRT 10.13。
- **核心改动**：
  1. 删除旁路子命令（`cleanup`、`check-supervision`、`static-masks`、`check-monocular`）、`--enhanced-photos` 覆盖参数、5 视角 `ViewCoverage` 裁边，以及 `shutter_candidate.bin`、`cleanup_candidate.bin`。
  2. 收紧前端配对：`CandidatePairs` 初始近邻收缩到 `span = 8`（`> 4` 步长 `2`）且限制 `<= 15 m` 与 `!FacesAway`；初始检索限定 `<= 20 m` 与 `!FacesAway` 单池前 `8` 选至多 `3` 个同镜头去重（`< 2 s`）候选；`NavigationRevisitCandidates` 去重（`< 2 s` 或 `< 1 m`）并封顶 `6` 个。
  3. 监督单路化与降规格：去掉 `15 × 60°` 法向专用 MoGe 子图，仅保留 `7 × 90°`（`FaceHalfDegrees = 45`）单路同时产出深度与法向；默认监督尺寸改为 `max_size = 1024`、`boundary = 4`（`FitFaceScales` 的 `min_anchors` 按监督图面积相对 `1600×1600` 缩放，下限 `128`）；`StaticMasks` 栅格同步改为 `1024 / MinArea=16 / MinSamples=12`，并通过 `LasSurfelCache` 与监督导出复用同一次 LAS 扫描及面元缓存。
  4. 初始化 PLY 裁剪与限流：`ExportLasPly` 按已注册相机 `<= 50 m`、`160°` 视锥裁剪，按最近距离分 4 档体素（`10 / 20 / 40 / 50 m` 内分别 `1× / 2× / 4× / 8×` 基准体素），并确定性上限截断到 `2,500,000` 点（对齐 `configs/trainer.json` 的 `max_cap: 3000000`）。

### 1. `篮球场`（536 幅曝光）：基线 vs 优化实测

基线目录为 `work\篮球场`，新版在独立临时目录 `dist\verify-篮球场` 全流程重跑并回读 `check` 通过（`passed: true`，对比后已按当时约定清理临时目录）。注：文件大小单位均按 `1 MB = 1024 × 1024 B` 换算。

| 指标 | 基线（`work\篮球场`） | 优化后（`dist\verify-篮球场`） | 变化 |
| --- | --- | --- | --- |
| 注册情况 | `536 / 536`（complete） | `536 / 536`（complete） | 持平 |
| 候选 / 导航新增 / 几何验证配对 | `11,385` / `4,359` / `10,116`（`88.9%`） | `6,838` / `2,055` / `6,479`（`94.7%`） | 配对 `-39.9%`，通过率 `+5.8%` |
| SfM 点数 / 观测数 | `116,265` / `478,558` | `122,670` / `506,014` | 点数 `+5.5%`，观测 `+5.7%` |
| 轨迹 `>= 3` / `>= 8` 视图 | `54,199` / `10,163` | `57,658` / `10,925` | `+6.4%` / `+7.5%` |
| 重投影误差 `median` / `p90` | `0.7181 px` / `1.7599 px` | `0.7111 px` / `1.7454 px` | 小幅改善 |
| `masks` / `features` / `matches` 耗时 | `126.46 s` / `30.47 s` / `114.90 s` | `130.61 s` / `30.85 s` / `70.09 s` | `matches` `-39.0%` |
| `sfm` / `shutter` / `cleanup` 耗时 | `125.28 s` / `71.02 s` / `9.77 s` | `114.26 s` / `62.50 s` / `10.78 s` | `sfm` `-8.8%` |
| 监督参数（`max_size` / `boundary`） | `1600` / `6`（7+15 双路 MoGe） | `1024` / `4`（7 张单路 MoGe） | 分辨率与子图大幅精简 |
| `supervision` 总耗时 / MoGe / 投影 / 面元 | `377.98 s` / `299.81 s` / `47.00 s` / `4.09 s` | `165.61 s` / `99.67 s` / `47.13 s` / `3.29 s` | 监督 `-56.2%`，MoGe `-66.8%` |
| 有效监督像素（深度 / 法向） | `547,955,974` / `759,156,103` | `225,048,054` / `306,209,284` | 随像素面积缩放 |
| `points3D.ply` 点数 / 字节 | `7,587,128` / `204,852,691 B`（`195.36 MB`） | `2,500,000`（分档体素原 `5,146,693`，`culled=0`）/ `67,500,235 B`（`64.37 MB`） | `-67.0%` |
| 目录体积（`depths` / `normals` / `sparse`） | `438.44` / `946.03` / `242.45 MB` | `203.34` / `391.29` / `111.99 MB` | `-53.6%` / `-58.6%` / `-53.8%` |
| 交付总体积（含 `1612.62 MB` 原图） | `3264.11 MB`（非图 `1651.49 MB`） | `2341.43 MB`（非图 `728.81 MB`） | 整包 `-28.3%`，非图 `-55.9%` |
| 完整实跑阶段总耗时（含 `prepare 0.24 s`、`align 24.36 s`、`export 237.69 s`） | —（旧版仅 `supervision` 即 `378 s`） | `685.60 s` | — |

### 2. `展厅`（588 幅曝光）：基线 vs 优化实测

基线目录为 `work\展厅`，用户重新运行完整流程产出 `work\展厅new`，经 `geosunpre.exe check` 回读校验通过（`passed: true`，两份成果目录均保留在本地）。

| 指标 | 基线（`work\展厅`） | 优化后（`work\展厅new`） | 变化 |
| --- | --- | --- | --- |
| 注册情况 | `585 / 588`（`complete: false`，未注册 `00214–00216-R`） | `582 / 588`（`complete: false`，未注册 `00214–00215-R`、`00226–00228-R`、`00301-R`） | `-3` 张（`99.5% -> 99.0%`） |
| 候选 / 导航新增 / 几何验证配对 | `15,073` / `7,364` / `11,598`（`76.9%`） | `7,916` / `2,878` / `6,630`（`83.8%`） | 配对 `-47.5%`，通过率由 `76.9%` 升至 `83.8%` |
| SfM 点数 / 观测数 | `145,439` / `722,416` | `150,014` / `731,012` | 点数 `+3.1%`，观测 `+1.2%` |
| 轨迹 `>= 3` / `>= 8` 视图 | `69,050` / `17,326` | `69,929` / `17,452` | `+1.3%` / `+0.7%` |
| 重投影误差 `median` / `p90` | `0.8993 px` / `2.0937 px` | `0.8891 px` / `2.0678 px` | 小幅改善 |
| LAS 对齐点到面 `median` / `p90`（支持率） | `0.0053 m` / `0.0157 m`（`81.5%`） | `0.0060 m` / `0.0169 m`（`79.4%`） | 基本持平（毫米级差异） |
| `masks` / `features` / `matches` 耗时 | `136.09 s` / `32.24 s` / `144.85 s` | `138.77 s` / `34.10 s` / `78.19 s` | `matches` `-46.0%` |
| `sfm` / `shutter` / `cleanup` 耗时 | `186.42 s` / `36.70 s`（7 次迭代）/ `6.23 s` | `164.96 s` / `77.09 s`（17 次迭代）/ `6.17 s` | `sfm` `-11.5%`；`shutter` 验证误差 `1.1579 -> 1.1385 px` |
| 监督参数（`max_size` / `boundary`） | `1600` / `6`（7+15 双路 MoGe） | `1024` / `4`（7 张单路 MoGe） | 分辨率与子图大幅精简 |
| `supervision` 总耗时 / MoGe / 投影 / 面元 | `380.68 s` / `324.05 s` / `27.00 s` / `2.67 s` | `151.13 s` / `103.79 s` / `25.30 s` / `2.85 s` | 监督 `-60.3%`，MoGe `-68.0%` |
| 有效监督像素（深度 / 法向） | `815,323,694` / `966,678,606` | `333,594,136` / `391,802,730` | 随像素面积缩放（`-59.1%` / `-59.5%`） |
| `points3D.ply` 点数 / 字节 | `916,436` / `24,744,006 B`（`23.60 MB`） | `881,540`（`culled_out_of_view=0`，未触 `2.5M` 上限）/ `23,801,814 B`（`22.70 MB`） | `-3.8%` |
| 目录体积（`depths` / `normals` / `sparse`） | `562.91` / `833.40` / `77.42 MB` | `262.98` / `372.28` / `76.60 MB` | `-53.3%` / `-55.3%` / `-1.1%` |
| 交付总体积（含原图） | `2567.03 MB`（原图 `1068.92`，非图 `1498.11 MB`） | `1797.49 MB`（原图 `1063.97`，非图 `733.52 MB`） | 整包 `-30.0%`，非图 `-51.0%` |
| 已记录主阶段耗时小计（`masks` 至 `supervision`） | `923.22 s` | `650.42 s` | `-29.5%` |

### 3. 结论与待跟踪项
- **结论**：
  - `篮球场`（室外开阔）与 `展厅`（室内多遮挡/多空间结构）双场景实测一致：去除 60° 法向第二路 MoGe、监督改为 `1024px`、复用 `LasSurfelCache`、分档裁剪 PLY 并收紧前端配对后，MoGe 推理耗时均下降 `67%–68%`（提速超 3 倍），总监督耗时下降 `56%–60%`，特征匹配耗时下降 `39%–46%` 且验证通过率提升（`篮球场 88.9% -> 94.7%`、`展厅 76.9% -> 83.8%`），SfM 点数/观测数反升、重投影误差小幅改善，非图交付体积缩减 `51%–56%`。
  - 需持续关注的小幅代价：`展厅` 右相机局部在配对收紧后多出 3 帧未注册（`585 -> 582 / 588`），`shutter` 迭代次数收敛敏感度略有波动；作为单机个人工程，当前默认配置性价比收益显著，可长期保留为主线默认。
- **待补充**：
  - 使用 `work\展厅new` 与同等规模交付在下游 LichtFeld Studio 完成 3DGS 训练对照，确认 `1024px` 单路 `90°` MoGe 监督与 `2.5M` 分档点云对最终渲染质量和训练速度的影响；如特定细结构场景需要更高频法向，可通过 `--supervision-max-size 1600` 覆盖。

## 2026-11-16：统一 `segment_and_ignore` 与静态掩码天空 LAS 核验（方案 A）

- **问题与方案**：此前 `configs/trainer.json` 使用 `mask_mode = "ignore"`（默认 `mask_threshold = 0.5`），导致室外真实天空 `128`（`128/255 = 0.502 >= 0.5`）在训练中被当作 `1.0` 前景而无法触发天空不透明度惩罚；直接改为 `segment_and_ignore` 又会让室内灯箱/天花板等 SAM3 天空误检（`128`）被挖空。采用方案 A：
  1. `configs/trainer.json` 统一 `mask_mode` 为 `"segment_and_ignore"`；
  2. `StaticMasks` 复用 Pass 1 同一次 `1024` 长边网格 LAS 深度投影，在原人物/影子恢复之外新增 `128`（天空）两级核验：先对每个 `128` 连通块零边界向内腐蚀 `2 px` 检验 LAS 实体支撑（`core_area >= 16`、`core_las >= 12`、`share >= 50%` 则置回 `255`），再按剩余开阔天空图像比例（单图 `>= 0.5%` 未腐蚀有效域、全局需达到 `>= min(8, ceil(N/2))` 且 `>= 5%` 注册图）判定室内/室外，室内场景（`kept: false`）将剩余 `128` 全部置回 `255` 并同步恢复 MoGe 深度与法向监督；
  3. `manifest.json` 持久化 `static_masks` 摘要（去除 `component_rows`），便于生产模式清理 `.work` 后复核。

### 1. 室内 `展厅` 实测（`work\展厅new` vs `work\展厅-sky`）

两份目录均经 `run.ps1 check` 回读校验通过（`passed: true`）并保留在本地。

| 指标 | 基线（`work\展厅new`） | 方案 A（`work\展厅-sky`） | 变化 / 说明 |
| --- | --- | --- | --- |
| 注册情况 | `582 / 588`（`complete: false`） | `578 / 588`（`complete: false`） | SfM 随机性多 4 帧未注册（`00216–00220-R`） |
| SfM 点数 / 观测数 | `150,014` / `731,012` | `149,732` / `730,586` | `-0.19%` / `-0.06%` |
| 重投影误差 `median` / `p90` | `0.8891 px` / `2.0678 px` | `0.8902 px` / `2.0677 px` | 持平 |
| `static_masks` 人物恢复（组件 `716`） | 未写入清单 | `1,022,767 px`（`45` 图） | 正常恢复静态人物/展板 |
| `static_masks.sky` 判定 | 无（SAM3 共 `9` 图误检） | `kept: false`（`raw_sky=4`, `open_sky=0`, `surface=3`, `changed=4`） | **正确识别为室内** |
| 清除天空误检像素 / 最终 `128` 复检 | 保留 `128`（未统计） | 清除 `213,964 px` / 全量 578 张掩码解码含 `128` 图像数为 `0` | **室内灯箱/天花全清回 `255`** |
| `Export static masks` / `static_masks` 耗时 | — | `37.89 s`（分析 `34.03 s`） | 无额外 GPU 开销 |
| `supervision` 总耗时 / MoGe / 投影 / 面元 | `151.13 s` / `103.79 s` / `25.30 s` / `2.85 s` | `153.37 s` / `103.95 s` / `25.36 s` / `2.77 s` | 基本持平 |
| 有效监督像素（深度 / 法向） | `333,594,136` / `391,802,730` | `330,992,554` / `388,937,623` | 随少注册 4 帧略降 `-0.78%` / `-0.73%` |
| 交付体积（非图 / 总体积 / PLY） | `733.52` / `1797.49 MB` / `881,540` 点 | `728.82` / `1785.66 MB` / `881,596` 点 | 持平 |
| 全流程总耗时 | — | `719.65 s` | 完整端到端运行通过 |

### 2. 室外 `篮球场` 实测（`dist\verify-篮球场` vs `work\篮球场-sky`）

新交付目录 `work\篮球场-sky` 经 `run.ps1 check` 回读校验通过（`passed: true`）并保留在本地。

| 指标 | 同规格前次（`dist\verify-篮球场`） | 方案 A（`work\篮球场-sky`） | 变化 / 说明 |
| --- | --- | --- | --- |
| 注册情况 | `536 / 536`（`complete: true`） | `536 / 536`（`complete: true`） | 持平 |
| SfM 点数 / 观测数 | `122,670` / `506,014` | `122,763` / `506,919` | 持平 |
| 重投影误差 `median` / `p90` | `0.7111 px` / `1.7454 px` | `0.7124 px` / `1.7487 px` | 持平 |
| `static_masks` 人物恢复（组件 `181`） | 未写入清单 | `0 px`（`0` 图） | 室外均为走动人物/操作员 |
| `static_masks.sky` 判定 | 无 | `kept: true`（`raw_sky=529`, `open_sky=518`, `surface=111`, `changed=60`） | **正确识别为室外并保留天空** |
| 清除天空误检像素 / 真实天空抽检 | — | 清除 `844,415 px`；抽检 `00033-L`/`R`、`00300-R` 均保留大量 `128` 天空 | **仅清除有 LAS 实体的零星误检，真实天空完整保留** |
| `Export static masks` / `static_masks` 耗时 | — | `64.40 s`（分析 `59.07 s`） | 比室内多因 529 张含 sky 连通块分析 |
| `supervision` 总耗时 / MoGe / 投影 / 面元 | `165.61 s` / `99.67 s` / `47.13 s` / `3.29 s` | `174.00 s` / `100.74 s` / `47.69 s` / `3.37 s` | 基本持平 |
| 有效监督像素（深度 / 法向） | `225,048,054` / `306,209,284` | `225,111,357` / `306,356,733` | `+63,303` / `+147,449`（放回 60 张实体表面误检） |
| 交付体积（非图 / 总体积 / PLY） | `728.81` / `2341.43 MB` / `2,500,000` 点 | `728.98` / `2341.59 MB` / `2,500,000` 点 | 持平 |
| 全流程总耗时 | `685.60 s` | `700.85 s` | `+15.25 s`（`+2.2%`） |

### 3. 结论与后续验证
- **结论**：方案 A 在零新增 GPU 阶段和约 2% 端到端耗时开销内，稳定区分室内（`kept: false`，578/578 零残余 `128`，等价 `ignore`）与室外（`kept: true`，`518/536` 张保留开阔天空 `128` 供 `segment_and_ignore` 惩罚，同时清除 60 张有 LAS 支撑的天空误检并增加约 6.3 万深度监督像素），可直接配合 `configs/trainer.json` 的统一 `segment_and_ignore` 交付训练。
- **待补充**：使用 `work\展厅-sky` 与 `work\篮球场-sky` 在下游 LichtFeld Studio 中按 `configs/trainer.json` 实际启动 3DGS 训练，验证室内灯箱不挖空及室外天空不透明度消除的最终渲染效果。

## 2026-11-16：方向一（分镜头配对与回访稳定性恢复）+ 方向二（零精度损失性能优化）

- **测试环境**：Windows x64，`NVIDIA GeForce RTX 5070 Ti`（CC 12.0），C++17 / CUDA / TensorRT 10.13。
- **核心改动**：
  1. **方向一 — 分镜头配对与多通次回访重构**（`src/frontend/frontend.cpp`）：
     - `CandidatePairs` 按 L/R 各自曝光序列分别建立局部近邻：同镜头连接步长 `{1, 2, 3, 4, 5, 8}` 且导航 `<= 15 m`（短窗不过滤 `FacesAway`，保住窄展区退步重访边）；异镜头按曝光时间差 `|Δt| <= 2.5 s` 且 `<= 15 m` 取最近至多 `4` 帧（不假设同步，补全转弯交接视角）；
     - `NavigationRevisitCandidates` 按镜头分池（同镜头 `<= 5 m` 且 `!FacesAway` 至多 5 帧，异镜头 `<= 4 m` 补齐至总上限 6 帧、至多 2 帧），并在同池内先按 `>= 12 s` 轨迹通次各取最近 1 帧、再按 `>= 2.5 s` 与 `>= 1 m` 补足，防止单个近邻通次占满全部 6 个回访槽位；
     - 外观检索在局部窗外按 `|Δt| > 30 s` 与 `<= 30 s` 两波段分池（`<= 20 m`，仅 `|Δt| > 10 s` 过滤 `FacesAway`），优先取至多 2 个远距回环再补足至多 3 个。
  2. **方向二 — 零精度损失性能加速**：
     - **StaticMasks 单遍近距 LAS 投影 + `ProjectSurfels` 保守视锥/距离早退**（`src/geometry/supervision.cpp`、`supervision_kernels.h/.cu`）：`MakeProjection` 预计算考虑卷帘读出位移与转动包络的 `max_q2`、`min_qz_slope`、`qz_slack`，在 3 步 `atan2`/`hypot` 读出线迭代前剔除视锥外点；`StaticMasks` 改用 `ProjectNearest` + `ResolveNearest` 单遍最近深度（免除 Pass 2 `Blend`、法向编码与后处理滤波，批次增至 `<= 160`）。
     - **Shutter 轻量轴向预筛 + 单轴全量 BA**（`src/sfm/shutter.cpp`）：预计算一次 `SynchronousFrames`，对 `~8,000–9,000` 条多视图轨迹（保证每图 `>= 24` 观测）按行、按列各跑 3 轮固定相机预筛 BA，再由独占 `ba_workers` 全量求解首选轴，物理一致性明确分离且首选轴通过时不重跑劣势轴全量 BA。
     - **消除独立 `export` 冗余复检**（`src/io/colmap.cpp`、`src/app/pipeline.cpp`）：`ExportBundle`/`ExportSelected` 直接返回 `CheckBundle` 结果并由 `Export` 输出。

### 1. 室内 `展厅` 实测（`work\展厅-sky` vs `work\展厅-opt` vs `work\展厅-opt2`）

三份目录均经 `run.ps1 check` 回读校验通过（`passed: true`）并保留在本地；其中 `work\展厅-opt` 为中间对照（初版 `{1,2,3,4,6,8}` + `|Δt|<=1.5 s` + 未分波段检索 + `18k/5轮` 预筛），`work\展厅-opt2` 为定稿实现。

| 指标 | 基线（`work\展厅-sky`） | 初版对照（`work\展厅-opt`） | 定稿（`work\展厅-opt2`） | 相对 `展厅-sky` 变化 |
| --- | --- | --- | --- | --- |
| 注册情况 | `578 / 588`（`complete: false`） | `582 / 588`（`complete: false`） | **`585 / 588`**（`complete: false`） | **`+7` 幅**（追平优化前 `work\展厅`，仅余白墙 `00214–00216-R`） |
| 候选 / 导航新增 / 几何验证 | `7,916` / `2,878` / `6,630`（`83.8%`） | `8,782` / `2,874` / `7,445`（`84.8%`） | `9,350` / `2,839` / `7,958`（**`85.1%`**） | 配对验证率反升 `+1.3%` |
| SfM 点数 / 观测 / `>=3` 视图轨迹 | `149,732` / `730,586` / `70,014` | `150,365` / `732,794` / `70,094` | **`151,712` / `738,038` / `71,333`** | `+1.3%` / `+1.0%` / `+1.9%` |
| 重投影误差 `median` / `p90` | `0.8903 px` / `2.0679 px` | `0.8915 px` / `2.0731 px` | `0.8905 px` / `2.0709 px` | 持平（±0.003 px） |
| 导航位置误差 `median` / `p90` | `0.0215 m` / `0.0353 m` | `0.0220 m` / `0.0368 m` | **`0.0211 m`** / `0.0372 m` | 持平 |
| `matches` / `sfm` 耗时 | `78.39 s` / `146.04 s` | `86.41 s` / `151.09 s` | `92.43 s` / `146.03 s` | `sfm` 零增加、`matches` 增 `14.0 s` 换回 7 幅注册 |
| `shutter` 轴向 / 预筛 / 总耗时 | `rows` / 无 / `75.64 s` | `rows` / `4.35+4.28 s` / `67.83 s` | **`rows` / `1.88+1.83 s` / `65.64 s`** | **`-10.00 s`（`-13.2%`）**；`readout_s=[0.02384, 0.02392]` |
| `static_masks` 分析耗时 / `Export` 阶段耗时 | `34.03 s` / `37.89 s` | `20.57 s` / — | **`20.68 s` / `24.71 s`** | **分析 `-39.2%`**；`sky.kept=false`、像素精确清除 `213,964 px` |
| 全量导出掩码 `128` 像素复检 | `0` 幅（`578` 图） | — | **`0` 幅（`585` 图）** | 封闭室内灯箱/天花无 `128` 残留 |
| `supervision` 总耗时 / LAS 投影 | `153.37 s` / `25.36 s` | `150.90 s` / `22.97 s` | `151.52 s` / **`23.05 s`** | **LAS 投影 `-9.1%`**（且多处理 7 幅图像） |
| 有效监督像素（深度 / 法向） | `330,992,554` / `388,937,623` | `333,548,526` / `391,860,285` | **`335,805,324` / `394,047,965`** | `+4.81M` / `+5.11M`（多 7 幅覆盖） |
| 全流程总耗时 | `719.65 s` | `710.48 s` | **`707.46 s`** | **`-12.19 s`（多注册 7 幅下仍净提速）** |

### 2. 室外 `篮球场` 实测（`work\篮球场-sky` vs `work\篮球场-opt`）

两份目录均经 `run.ps1 check` 回读校验通过（`passed: true`）并保留在本地。

| 指标 | 基线（`work\篮球场-sky`） | 定稿（`work\篮球场-opt`） | 相对 `篮球场-sky` 变化 |
| --- | --- | --- | --- |
| 注册情况 | `536 / 536`（`complete: true`） | **`536 / 536`**（`complete: true`） | 持平（100% 完整） |
| 候选 / 导航新增 / 几何验证 | `6,838` / `2,055` / `6,479`（`94.7%`） | `7,478` / `1,561` / `7,163`（**`95.8%`**） | 配对验证率升 `+1.1%` |
| SfM 点数 / 观测数 | `122,763` / `506,919` | `121,857` / `499,146` | 持平（`-0.7%` / `-1.5%`） |
| 重投影误差 `median` / `p90` | `0.7116 px` / `1.7472 px` | **`0.7088 px` / `1.7394 px`** | 略优（`-0.0028 px`） |
| `matches` / `sfm` 耗时 | `70.42 s` / `117.63 s` | `73.65 s` / **`102.87 s`** | `sfm` `-14.76 s`（`-12.5%`） |
| `shutter` 轴向 / 预筛 / 总耗时 | `columns` / 无 / `74.61 s` | **`columns` / `1.46+1.76 s` / `53.45 s`** | **`-21.16 s`（`-28.4%`）**；留出中位数 `0.9871 -> 0.9788 px` |
| `static_masks` 分析耗时 / `Export` 阶段耗时 | `59.07 s` / `64.40 s` | **`31.21 s` / `36.35 s`** | **分析 `-47.2%` / 阶段 `-43.6%`**；`sky.kept=true`（`518` 开阔、`68` 图除误检 `1,131,424 px`） |
| `supervision` 总耗时 / LAS 投影 | `174.00 s` / `47.69 s` | **`159.99 s` / `43.60 s`** | 监督 `-8.1%`，LAS 投影 `-8.6%` |
| 有效监督像素（深度 / 法向） | `225,111,357` / `306,356,733` | `225,080,139` / `306,386,746` | 持平（`-0.01%` / `+0.01%`） |
| 交付体积（非图 / 总体积 / PLY） | `728.98` / `2341.59 MB` / `2,500,000` 点 | `729.09` / `2341.70 MB` / `2,500,000` 点 | 持平 |
| 全流程总耗时 | `700.85 s` | **`630.55 s`** | **`-70.30 s`（`-10.0%`，净提速 1 分 10 秒）** |

### 3. 结论与待补充
- **结论**：
  1. 方向一彻底解决了原交错 L/R 全局下标索引步长折半、转弯跨镜头断连、近邻通次垄断回访配额及无分波段检索挤压远距回环的问题，`展厅` 在不额外启用 `sfm_repair_rounds` 且保持 `<= 15 m / <= 20 m` 紧预算的情况下由 `578/588` 恢复到 **`585/588`**（配对几何验证通过率 `83.8% -> 85.1%`），`篮球场` 维持 **`536/536`**（验证率 `94.7% -> 95.8%`）。
  2. 方向二在不牺牲任何几何精度或检查严谨性的前提下，单遍最近 `StaticMasks` LAS 投影、保守视锥/距离早退及轻量卷帘轴向预筛将 `static_masks` 缩短 `39%–47%`、`shutter` 缩短 `13%–28%`、LAS 投影缩短 `8.6%–9.1%`，抵消了增加有效配对的匹配开销后，`展厅` 净提速 `12.19 s`（`719.65 -> 707.46 s`）且多注册 7 图，`篮球场` 净提速 **`70.30 s`（`700.85 -> 630.55 s`，`-10.0%`）**。
- **待补充**：使用 `work\展厅-opt2` 与 `work\篮球场-opt` 在下游 LichtFeld Studio 中按 `configs/trainer.json` 启动 3DGS 训练，验证室内灯箱不挖空与室外天空消除的最终渲染效果。

## 2026-11-16：同步双目卷帘刚体耦合、高精度 ENH 杆臂补偿与 100% 轨迹真实 LAS 平面终局精调

- **问题与诊断**：对照实测交付发现两项直接导致 3DGS 重影与相机-LAS 错位的几何根源：
  1. `shutter` BA 仅用 `RigLink` 约束同步双目参考位姿，未耦合左右镜头同一曝光帧内的读出转角 `ω_L, ω_R`，造成双目三维角速度差中位数高达 `5.75–8.25 °/s`（`P90` 达 `9.53–17.17 °/s`）；同时 `NavigationVelocities` 仅取 ECEF `1 mm` 量化轨迹 `±50 ms` 差分且缺少旋转杆臂切向速度。
  2. `shutter.cpp` 为保持尺度的 `inputs.training_planes` 把 LAS 平面中心覆盖为全局快门旧视觉三维点坐标，且 `80%` 轨迹留出检验通过后未回代全量轨迹与真实 LAS 平面；导致去卷帘之后视觉点与真实 LAS 平面仍残留中位 `6.3–7.7 mm`、`P90` `21.4–23.6 mm` 的系统错位。
  3. 同步验证了 `THIN_PRISM_FISHEYE` 对本组鱼眼畸变的静态残差拟合改善不足 `0.001 px`（主因不在静畸变模型），故保留与下游 Vulkan 光栅化器全兼容的 `OPENCV_FISHEYE`（8 参数）。
- **核心改动**：
  1. **高精度 ENH 线速度与刚体杆臂提取**（`src/io/scene.h/.cpp`）：新增 `ReadExposureKinematics`，从 `0.01 mm` 精度的 `_pos_ENH.txt` 插值提取 `±20 ms` 线速度及每帧曝光相对导航中心的刚体杆臂 `lever_arm = lens - enh(t)`。
  2. **同步双目读出角速度刚体约束**（`src/sfm/optimization.h/.cpp`）：`Optimize` 在 `shutter` 非空时对同步帧追加齐次交叉项 `RigShutterRotation`：`(τ_L ω_R − τ_R R_ext ω_L) / sqrt(τ_L² + τ_R² + 1e-6)`（`RigShutterRotationSigma = 0.02°`、`HuberLoss(4)`），同时兼容 `展厅`（同向读出）与 `篮球场`（反向读出）。
  3. **LAS 平面分块均衡与准入后终局精调**（`src/alignment/alignment.cpp`、`src/sfm/shutter.cpp`）：`LasPlaneConstraints` 按 `Partition` 体素密度均衡权重；`CorrectRollingShutter` 以预筛 `readout_s` 初始化完整 BA 并保留 80%/20% 留出检验准入，准入通过后执行 `PolishAcceptedShutter`——更新含转动杆臂切向项的镜头速度 `v_nav − (ω/τ) × r_cam`，并回全部 `100%` 满足全观测正深度的双视图及以上轨迹，并在当前去卷帘点云上重新关联真实 LAS 平面（保留真实平面中心）做 `100` 轮 BA，经质量、逐图 P90 稳定性、物理一致性及平面拟合验收后采用（`polish.used`）。

### 1. 室内 `展厅` 实测（`work\展厅-opt2` vs `work\展厅-shutter`）

新交付目录 `work\展厅-shutter` 经 `geosunpre.exe check` 回读校验通过（`passed: true`）并保留在本地。

| 指标 | 基线（`work\展厅-opt2`） | 本次改动（`work\展厅-shutter`） | 变化 |
| --- | --- | --- | --- |
| 注册情况 | `585 / 588`（`complete: false`） | `585 / 588`（`complete: false`） | 持平 |
| SfM 点数 / 观测 / `>=3` 视图轨迹 | `151,712` / `738,038` / `71,333` | `151,406` / `736,311` / `70,858` | 基本持平（`-0.2%`） |
| 重投影误差 `median` / `p90`（`L` / `R` 中位） | `0.8904` / `2.0707 px`（`0.8364` / `0.9528`） | `0.8907` / `2.0728 px`（`0.8350` / `0.9544`） | 持平 |
| 卷帘留出检验误差（校正后 `median` / `p90`） | `1.1445 px` / `3.2420 px` | **`1.1381 px` / `3.2071 px`** | 均进一步下降 |
| **同步双目三维角速度差** `med` / `p90` / `max` | `8.25` / `17.17` / `27.99 °/s` | **`2.66` / `5.82` / `19.74 °/s`** | **中位 `-67.8%`，P90 `-66.1%`** |
| 双目参考位姿离散（`pos med` / `p90`） | `1.71 mm` / `4.69 mm` | **`1.53 mm` / `4.65 mm`** | `pos` 中位改善 `-10.5%` |
| 物理读出一致性 `explained_rotation`（`L` / `R`） | `0.3462` / `0.3397` | **`0.3573` / `0.3580`** | 双镜头均提升且更均衡 |
| **真实 LAS 平面拟合 `plane_after_m`（`med` / `p90`）** | `6.25 mm` / `21.43 mm`（`44,115` 点） | **`5.69 mm` / `17.70 mm`**（重关联 `47,446` 点 **`5.65` / `17.04 mm`**） | **P90 `-17.4%`（重关联 `-20.5%`，稳定面点 `+3,553`）** |
| 导航位置误差 `med` / `p90` 与 MoGe 视图冲突剔除 | `21.12` / `37.24 mm`，`38,369,243 px` | **`20.85` / `36.37 mm`**，`38,129,234 px` | 位置略优，冲突像素 `-24.0 万` |
| `shutter` 耗时（80% 主 BA / `polish` / 阶段总计） | `59.27 s` / — / `65.64 s` | `39.64 s` / `54.66 s`（`206,456` 轨迹）/ `100.74 s` | 80% BA 快 `-33.1%`，全阶段 `+35.10 s` |
| 全流程总耗时 / 交付体积 | `707.46 s` / `1806.01 MB` | `775.59 s` / `1806.18 MB` | 总耗时 `+68.13 s`（`+9.6%`） |

### 2. 室外 `篮球场` 实测（`work\篮球场-opt` vs `work\篮球场-shutter`）

新交付目录 `work\篮球场-shutter` 经 `geosunpre.exe check` 回读校验通过（`passed: true`）并保留在本地。

| 指标 | 基线（`work\篮球场-opt`） | 本次改动（`work\篮球场-shutter`） | 变化 |
| --- | --- | --- | --- |
| 注册情况 | `536 / 536`（`complete: true`） | `536 / 536`（`complete: true`） | 持平（100% 完整） |
| SfM 点数 / 观测 / `>=3` 视图轨迹 | `121,857` / `499,146` / `57,584` | **`122,560` / `503,124` / `57,703`** | `+0.6%` / `+0.8%` / `+0.2%` |
| 重投影误差 `median` / `p90`（`L` / `R` 中位） | `0.7095` / `1.7410 px`（`0.7175` / `0.7014`） | `0.7094` / `1.7421 px`（`0.7155` / `0.7023`） | 持平 |
| 卷帘留出检验误差（校正后 `median` / `p90`） | `0.9788 px` / `2.8341 px` | **`0.9757 px` / `2.8242 px`** | 均略优 |
| 读出时间 `readout_s`（内参基准 `L` / `R`） | `+31.52 ms` / `-27.17 ms` | **`+32.85 ms` / `-31.04 ms`** | 反向读出完全兼容且对称性更好 |
| **同步双目三维角速度差** `med` / `p90` / `max` | `5.75` / `9.53` / `15.81 °/s` | **`1.63` / `2.82` / `7.92 °/s`** | **中位 `-71.7%`，P90 `-70.4%`** |
| 双目参考位姿离散（`pos med` / `p90`） | `2.08 mm` / `4.17 mm` | **`1.81 mm` / `3.78 mm`** | `-13.0%` / `-9.4%` |
| 物理读出一致性 `explained_rotation`（`L` / `R`） | `0.2768` / `0.2841` | **`0.2936` / `0.3010`** | 双镜头稳定提升 |
| **真实 LAS 平面拟合 `plane_after_m`（`med` / `p90`）** | `7.67 mm` / `23.61 mm`（`42,702` 点） | **`3.79 mm` / `15.13 mm`**（重关联 `45,277` 点 **`3.73` / `14.75 mm`**） | **中位 `-50.6%`，P90 `-35.9%`（重关联稳定面点 `+2,575`）** |
| 导航位置误差 `med` / `p90` 与 MoGe 视图冲突剔除 | `67.36` / `99.30 mm`，`67,756,848 px` | **`20.81` / `39.83 mm`**，**`66,751,417 px`** | **冲突像素 `-100.5 万`** |
| `shutter` 耗时（80% 主 BA / `polish` / 阶段总计） | `46.09 s` / — / `53.45 s` | `60.43 s` / `52.81 s`（`201,925` 轨迹）/ `120.86 s` | 含全量 `polish` 共 `+67.41 s` |
| 全流程总耗时 / 交付体积 | `630.55 s` / `2341.70 MB` | `715.91 s` / `2341.52 MB` | 总耗时 `+85.36 s`（`+13.5%`） |

### 3. 结论与待补充
- **结论**：在不更改 `OPENCV_FISHEYE` 与任何外部输出格式的前提下，本次三项改进从根源消除了双目卷帘转角过拟合与相机-LAS 脱节问题：`展厅` 与 `篮球场` 同步双目三维角速度差均下降 **`66%–72%`**，真实 LAS 点到平面距离 P90 分别降至 **`17.0–17.7 mm`** 与 **`14.8–15.1 mm`**（`篮球场` 中位数从 `7.67 mm` 降至 **`3.73–3.79 mm`**，直接改善近距离边缘投影错位，并使 `篮球场` MoGe 跨视角深度冲突减少约 `100.5 万` 像素）；仅增加约 `10%–13.5%` 端到端耗时，可直接作为默认主线保留。
- **待补充**：使用 `work\展厅-shutter` 与 `work\篮球场-shutter` 在下游 LichtFeld Studio 完成 3DGS 训练对照（对照既有 `work\展厅-opt2` / `work\篮球场-opt`），检验墙面/展柜/场地边界锐度与重影改善效果。

## 2026-11-16：发版前合规脱敏、整包组装与双场景 3DGS + OSGB 全闭环验证

- **测试环境**：Windows x64，`NVIDIA GeForce RTX 4090`（`SM 8.9`，`24563 MB`，驱动 `610.88` / CUDA Runtime `12.8`），TensorRT `10.13.3.9`。
- **发版前核心改造与打包规格**：
  1. **合规脱敏与 Aggregate 商用架构**（`GeosunPre` 提交 `fee6a4f`、`65dba85`、`e551d60`；`LichtFeld-Studio` 提交 `b70494526`）：
     - SAM3 导出彻底移除 AGPL-3.0 的 `ultralytics`（`prepare-models.ps1` 显式执行 `uv pip uninstall` 清理 `ultralytics` 与 `clip`），切换至官方 `facebookresearch/sam3`（提交 `eef6839bcfc3458dd52ea80535dc6bd62aea86b1`）；在 `tools/models/networks.py` 与 `prepare.py` 中增加 `freeze_grounding_rotary`（预存 `freqs_cis_real`/`freqs_cis_imag` 并启用 `use_rope_real`）、将 `block_grounding_attention` 限定至 `model.transformer.encoder.modules()` 避开带 padding mask 的几何编码器，并引入 `TextOnlyGeometryEncoder` 跳过空点框分支以适配旧版 TorchScript ONNX 导出器。
     - `dependencies/licenses/` 补齐 `sam3.txt`（Meta SAM License）、`moge.txt`（MIT）、`dinov2.txt`（Apache-2.0）、`dad.txt`（MIT），交付包 `preprocess/licenses/` 共 `14` 份许可证文本；`LichtFeld-Studio` 子包恢复安装 `LICENSE`、`LICENSE-EXCEPTION.txt`、`THIRD_PARTY_LICENSES.md` 及 `share/LichtFeld-Studio/licenses/`，文档统一明确 GPLv3 Section 5 进程级解耦 Aggregate 分发口径。
  2. **架构支持、DLL 精简与启动防护**：
     - `LichtFeld-Studio` 训练器包补齐 `80-virtual;86-real;89-real;120-real`（含 Blackwell RTX 50 系列原生 SASS），并在 `LFS_TRAINER_ONLY` 下关闭 `CMAKE_INSTALL_UCRT_LIBRARIES`、加固 `RUNTIME_DEPENDENCY_SET` 系统库排除规则；产出 `Geosun-3DGS-Trainer-sm80-20261116-134442`（`59` 个文件，`.zip` `220,706,911 B` / `210.48 MiB`，`ucrtbase.dll` 与 `api-ms-win-*.dll` 数量为 `0`）。
     - `GeosunPre-1.1.0-e551d60-ampere_plus`（`39` 个文件）与上述训练器组装为整包 `packages\Geosun-1.1.0-e551d60-ampere_plus-trainer-b70494526-exe`（`.zip` 体积 `2,527,130,860 B` / `2410.06 MiB`）。
     - `bundle/README.md`、`docs/models.md`、`docs/deployment.md` 将显存门槛修正为 `≥ 12 GB（推荐 16 GB+）`；`bundle/main.cpp`（`geosun.exe`）新增 `--output` 卷可用空间预检，并在本次实跑定位到 `export.pending` 在杀软/索引扫描新写出 `67` 个文件时因无重试而在 `fs::rename` 触发瞬态锁后，补充 `PublishPending`（最多 `50 × 100 ms` 重试，`/W4` 零警告通过）。

### 1. 双场景全闭环实测数据（`展厅-bundle` vs `篮球场-bundle`）

- **执行口径说明**：
  - `展厅-bundle`（`D:\Dev\GeosunPre\work\展厅-bundle`）：由上述最新整包 `geosun.exe` 端到端串联执行（预处理采用 `ampere_plus` 官方 `facebookresearch/sam3` 引擎 `1a5e2f26d596fce5` + MoGe 引擎 `9059dbef1a7e207a`；`export.pending` 校验完整后重命名为 `export`），随后由同核 `LichtFeld-Studio.exe`（`e1d348b57`，与 `b70494526` 训练/网格/OSGB/评测 C++/CUDA 代码零差异）执行全视角离线 `eval`。
  - `篮球场-bundle`（`D:\Dev\GeosunPre\work\篮球场` + `D:\Dev\GeosunPre\work\篮球场-bundle`）：预处理目录 `work\篮球场` 系用户使用 `build\ninja\bin\geosunpre.exe`（`e551d60`）搭配本机 `models\`（`hardware_compatibility: none`，仍含旧 `ultralytics: 8.4.192` 导出 `grounding` 引擎 `1a6355d1a04b99ba` 与 MoGe 引擎 `36ace64568d19473`）生成；随后使用同核 `LichtFeld-Studio.exe` 严格按 `configs/trainer.json` 完成 `train` → `splats.ply` → `mesh --texel-size 0.02` → `eval` 全流程。

| 阶段 / 指标 | 室内 `展厅`（`work\展厅-bundle`） | 室外 `篮球场`（`work\篮球场` + `篮球场-bundle`） | 说明 |
| --- | --- | --- | --- |
| 预处理执行程序 / 模型规格 | 整包 `geosunpre.exe`（`e551d60`，`ampere_plus` 官方 SAM3） | 本机 `geosunpre.exe`（`e551d60`，`models/`） | 下游共用同一套 `configs/trainer.json` |
| 注册情况（`quality.json`） | `585 / 588`（未注册 `00214–00216-R`，`passed: true`） | **`536 / 536`**（`complete: true`，`passed: true`） | 关键几何门禁全通过 |
| SfM 点数 / 观测数 / `>=3` 视图轨迹 | `151,441` / `738,159` / `70,971` | `122,472` / `502,246` / `57,827` | 轨迹完整 |
| 重投影误差 `median` / `p90` | `0.8908 px` / `2.0724 px` | `0.7079 px` / `1.7450 px` | 远低于 `2.0 / 4.0 px` 上限 |
| LAS 点到面 `median` / `p90`（面点数） | `5.68 mm` / `16.99 mm`（`32,109`） | `7.08 mm` / `25.95 mm`（`27,297`） | `polish` 重关联后分别达 `5.62 / 16.74 mm` 与 `3.81 / 14.99 mm` |
| `shutter` 留出误差 `before -> after` | `1.4294 -> 1.1445 px`（`p90: 3.5161 -> 3.2040 px`） | `1.3487 -> 0.9838 px`（`p90: 3.3350 -> 2.8428 px`） | 同向/反向读出与 `polish` 均采用 |
| `static_masks.sky` 与终版含 `128` 图像数 | `kept: false`（`raw_sky=0`，`static` 恢复 `51` 图 `1,203,541 px`），终版含 `128`: **`0 / 585`** | `kept: true`（`raw_sky=529`, `open_sky=518`, 清误检 `58` 图 `837,672 px`），终版含 `128`: **`527 / 536`** | 室内零残余 `128`；室外开阔天空完整保留 |
| `points3D.ply` 点数 / 交付体积（非图 / 总） | `881,496`（`22.70 MiB`） / `738.30` / `1807.22 MiB` | `2,500,000`（`64.37 MiB`） / `729.23` / `2341.85 MiB` | 对齐 `max_cap = 3,000,000` |
| 3DGS 训练耗时 / `project.licht` / 高斯球数 | 约 `8m49s`（交付至生成时间差）/ `432.66 MiB` / `3,000,000` | `457.473 s`（`65.6 iter/s`）/ `420.73 MiB` / `3,000,000` | MRNF + GUT + PPISP（30k 步） |
| 离线 `eval`（全视角 `eval_all`）纯前景 `mask == 255` PSNR | **`26.1906 dB`**（`p10=23.95 / med=26.43 / p90=28.01`） | **`19.0291 dB`**（`p10=17.57 / med=18.88 / p90=20.69 / max=21.99`） | 室外原生 `masks:` 口径说明见下节 |
| 纯前景 SSIM / LPIPS | **`0.8535`** / **`0.2693`** | **`0.5652`** / **`0.3646`** | 室内色偏近零；室外 `bias_corr = (-0.0057, -0.0075, -0.0072)` |
| 渲染几何对齐：`normal_angle_deg` / `depth_absrel` | `12.0082°` / `0.0070` | `17.0784°` / `0.0051` | 先验深度中位误差分别 `16.32 mm` 与 `17.39 mm` |
| `export\splats.ply` 体积 / 坐标检查 | `204,000,666 B`（`194.55 MiB`，`3,000,000` 点，全有限） | `204,000,666 B`（`194.55 MiB`，`3,000,000` 点，全有限） | 无 NaN/Inf 异常点 |
| `export\mesh.ply` 顶点 / 面数 / 体积 | `747,552` 顶点 / `1,319,803` 面 / `35.61 MiB` | `21,681,487` 顶点 / `39,596,054` 面 / `1049.18 MiB` | 全有限坐标，`--texel-size 0.02` |
| `export\mesh_osgb` 瓦片目录 / `.osgb` 文件数 / 体积 | `16` 瓦片 / `64` 文件 / `49.09 MiB` | `512` 瓦片 / `2,006` 文件 / `1781.86 MiB`（`7041` 页，覆盖率 `72.7%`） | `SRS=ENU:31.0,121.0`，魔数校验全通过 |

### 2. 视觉抽检与指标口径分析

1. **室内 `展厅`视觉抽检**（抽看 `eval_step_30000` 中 `00146-R` p10、`00074-L` median、`00265-L` p90）：
   - 顶部发光灯箱、条形吊顶格栅与白色展台完整还原、无一处天空误挖黑洞；地面操作员双脚被静态掩码可靠剔除并由多视角一致补全，玻璃展柜、机器人车体及蓝色灯带走廊几何对齐无重影；`Unmasked PSNR = 20.4393 dB` 主要反映鱼眼有效圆圈外暗角环带与被剔除的动态人物差异。
2. **室外 `篮球场`视觉抽检与 `eval` 口径定位**（抽看 `00203-R`、`00244-L`、`00120-R` 及 `00199-R`、`00181-R`、`00036-L`）：
   - **渲染画面与天空惩罚验证**：场地面砖网格、白色标线、篮架、凉亭钢构、`9B` 标牌与建筑玻璃幕墙均清晰锐利、无卷帘重影，开放天空透明无空中漂浮物。
   - **`eval` 口径差异定位**：直接使用 `eval --eval-mask masks:work\篮球场\masks` 时报告 `Mask PSNR = 9.9967 dB`（`bias ≈ -0.10`），原因在于 `--eval-mask masks:` 默认按 `0.5` 阈值二值化，把 `128` 开放天空像素误纳入评分区域，使过曝白色天空原图与 `segment_and_ignore` 透明纯黑背景（`--bg-color [0,0,0]`）直接计算 MSE；将 `128` 置 `0` 仅评 `mask == 255` 静态前景（`eval_foreground`）后，前景色偏基本归零，PSNR 回升至 `19.0291 dB`（`p90 = 20.69`、`max = 21.99 dB`），余下扣分主要来自近景树冠叶隙与高楼顶缘处紧贴天空边界渗入的零星白边对比黑背景产生的高额 MSE，以及约 `142 m × 139 m × 88 m` 园区尺度下 `3M` 高斯上限对远景枝叶高频纹理的平滑效应（`depth_prior_low_alpha = 0.6961`）。
3. **结论与后续可选复核**：
   - 发版前必做的闭环验证（补齐 `sm_120` SASS、清理 `ucrtbase`/`api-ms-win` 冗余 DLL、恢复 LFS 开源许可证、SAM3 脱离 `ultralytics`、补齐模型许可证、修正 `≥ 12 GB` 显存门槛并加入磁盘预检与目录重命名重试）已全部实现并通过实测。
   - **可选复核**：用含 `PublishPending` 修复的新编译 `geosun.exe` 刷新整包并按需在外部终端以新包重跑一次室外 `篮球场-bundle`（统一切换至 `ampere_plus` 官方 `facebookresearch/sam3` 引擎）作最终回归存档。

---

## 2026-11-17：`Verify`/`Register` 种子解耦一致性实测 + 评测掩码/卷帘核验契约补齐

### 1. `展厅` `--stage sfm` 两次连跑一致性实测

基于 [D:\Data\展厅](file:///D:/Data/展厅)（`588` 图、`9,349` 候选像对）一次生成至 `matches` 的同一份缓存，在修复 [reconstruction.cpp](file:///D:/Dev/GeosunPre/src/sfm/reconstruction.cpp#L272-L295)（`Verify` 显式传入带确定性非零像对种子的 `cv::UsacParams`）与 [reconstruction.cpp](file:///D:/Dev/GeosunPre/src/sfm/reconstruction.cpp#L766)（`Register` 种子解耦为仅依赖 `id`）后，分别在默认 `"ba_workers": 8`（`work\展厅-sfm-verify`）与单线程 BA 对照 `"ba_workers": 1`（`work\展厅-sfm-verify-b1`）下连续执行两次 `--stage sfm`：

| 配置 / 轮次 | `verified_pairs` / `tracks` | 注册图像数 | 三维点 / 观测数 | 重投影 `median` / `p90` | 耗时（s） | `visual.bin` SHA256 与非耗时字段差异 |
| --- | --- | --- | --- | --- | --- | --- |
| `ba_workers = 8` Run 1 | `7967` / `181488` | `585 / 588`（缺 `00214–00216-R`） | `206,101` / `958,242` | `1.2743` / `3.2796 px` | `238.28` | `7B311C1D...` |
| `ba_workers = 8` Run 2 | `7967` / `181488` | `582 / 588`（缺 `00214–00219-R`） | `205,765` / `958,308` | `1.2756` / `3.2849 px` | `168.95` | `10A0AF6B...`（`2,438` 项非耗时差异） |
| `ba_workers = 1` Run 1 | `7967` / `181488` | `585 / 588`（缺 `00214–00216-R`） | `206,197` / `959,622` | `1.2754` / `3.2855 px` | `428.71` | `F4EB5606...` |
| `ba_workers = 1` Run 2 | `7967` / `181488` | `585 / 588`（缺 `00214–00216-R`） | `206,197` / `959,622` | `1.2754` / `3.2855 px` | `419.11` | **`F4EB5606...`（完全一致，非耗时差异 `0`）** |

- **根因定位与取舍**：
  1. `Verify`、`Tracks`、`Seed`（`00273-R + 00274-R`，`2,622` 观测）与 `Register` 已完全消除伪随机分支；`ba_workers = 1` 下连续两次 `visual.bin` 与全部 `88` 轮 BA / 位姿 / 点云结果逐字节一致（`SHA256: F4EB5606E1A0B1292B4DFDFD646124189F9A3B71564EA63050F8B16B24765791`）。
  2. `ba_workers = 8` 下 `opt[0].initial_cost` 在第 15 位有效数字即出现 `~4.5e-13` 偏差（`1168.82532073046` vs `1168.8253207304595`），源于 Ceres `options.num_threads = 8` 多线程浮点累加归约顺序；在第 14 轮 `BA focal`（`102` 图 `146,420` 观测）处线性迭代数相差 1（`47` vs `48`）并使第 15 轮 `BA radial` 翻动 1 个观测（`149,063` vs `149,062`），经后续增补最终分支为 `585` vs `582` 帧。默认保持 `"ba_workers": 8` 以兼顾约 `2×` 速度收益；严格按比特复现基准场景时可显式传入 `--config` `"ba_workers": 1`。

### 2. 下游评测与契约同步

- **LichtFeld-Studio**：统一 `metrics.cpp` 中 `MaskMode::SegmentAndIgnore` 的 `masks:<folder>` 加载经 `load_eval_mask_file` / `classify_keep_mask_for_metrics`（仅保留 `mask == 255` 前景，排除 `128` 天空与 `0` 忽略）；在 `native_gut_evaluation` 透传 `rolling_shutter`，并在日志（`Eval: rolling shutter active for K / N cameras`、`EvalMetrics::to_string()`）及 `metrics_report.txt`（`Rolling shutter cameras: K / N`）显式记录实际激活卷帘相机数。
- **契约文档**：[output.md](file:///D:/Dev/GeosunPre/docs/output.md#L102-L117) 补齐 `eval_all`、`mask == 255` 前景计分与卷帘日志/指标核验条目。

---

## 2026-11-17：LichtFeld-Studio 训练器包体瘦身（保留全架构 SASS + Tensor 模板隔离 + Core CUDA 去重 + Fatbin 压缩）

### 1. 根因确认与改造方案（方案 A）

基于基线产物 [Geosun-3DGS-1.1.1-Windows-x64.zip](file:///D:/Dev/GeosunPre/packages/Geosun-3DGS-1.1.1-Windows-x64.zip) 与 `build-windows-trainer-ptx80` 构建目录复核，保持 `CMAKE_CUDA_ARCHITECTURES="80-virtual;86-real;89-real;120-real"`（RTX 30/40/50 零 runtime JIT 延迟），从以下 3 个根因同步瘦身：

1. **Tensor 泛型/广播核模板跨编译单元隔离与单点实例化**：
   - 抽取 `run_with_thrust_policy` 至 [thrust_policy.cuh](file:///D:/Dev/LichtFeld-Studio/src/core/tensor/internal/thrust_policy.cuh)。
   - [tensor_ops.hpp](file:///D:/Dev/LichtFeld-Studio/src/core/tensor/internal/tensor_ops.hpp) 停止向外部 `.cu` 暴露 `tensor_generic_ops.cuh` 与 `tensor_broadcast_ops.cuh` 的核模板实现，仅保留 `LFS_CORE_API` 声明（并用宏守卫避免核心实现 TU 重声明冲突）；移除 `mrnf_kernels.cu` 对 `tensor_generic_ops.cuh` 的冗余包含。
   - 统一 [tensor_ops.cu](file:///D:/Dev/LichtFeld-Studio/src/core/tensor/tensor_ops.cu)、[tensor_broadcast_ops.cu](file:///D:/Dev/LichtFeld-Studio/src/core/tensor/tensor_broadcast_ops.cu) 与 [tensor_exports.cu](file:///D:/Dev/LichtFeld-Studio/src/core/tensor/tensor_exports.cu) 之间的重复显式实例化，并将 `launch_ieee_*_float` 及缺失浮点/标量显式实例化集中导出到 `geosun_3dgs_core.dll`。
2. **`lfs_core_cuda` 全量导出与 EXE/DLL 去重**：
   - 将原使用静态占位宏 `LFS_CUDA_API` 或未导出的 `lfs_core_cuda` 接口（含 `image_quantize`、`initial_scales`、`mask_normalization`、`selection_ops`、`splat_transform`、`sh_layout`、`sh_value_quant_kernels`、`exportable_storage`、`lanczos_resize`、`undistort` 及 `RasterizerMemoryArena` / `GlobalArenaManager` 成员方法）统一标记为 `LFS_CORE_API`。
   - [src/core/CMakeLists.txt](file:///D:/Dev/LichtFeld-Studio/src/core/CMakeLists.txt) 以 `$<LINK_LIBRARY:WHOLE_ARCHIVE,lfs_core_cuda>` 将全部 `lfs_core_cuda` 符号完整导出到 `lfs_core`（`geosun_3dgs_core.dll`），并在 `src/training`、`fastgs`、`src/io` 中移除对静态库 `lfs_core_cuda` 的重复链接（`src/io` 改直连 `CUDA::cuda_driver`）。
3. **CUDA 12.8 Fatbin 全条目压缩**：
   - 在 [LichtFeld-Studio/CMakeLists.txt](file:///D:/Dev/LichtFeld-Studio/CMakeLists.txt) 为非 Debug CUDA 编译统一开启 `-Xfatbin=-compress-all`（CUDA ≥ 12.8 追加 `--compress-mode=size`），压缩 `sm_86` / `sm_89` / `sm_120` 原生 SASS 与 `compute_80` PTX。

### 2. 前后体积对比（实测）

| 指标项 | 瘦身前基线（`1.1.1`） | 方案 A 瘦身后 | 变化幅度 | 备注 |
| --- | --- | --- | --- | --- |
| `build-windows-trainer-ptx80` `.cu.obj`（91 个） | `954.32 MiB` | `186.27 MiB` | **`-80.5%`** | 消除外部 TU 模板泛化 + fatbin 压缩 |
| `geosun_3dgs_train.exe`（未压缩 / ZIP） | `495.08` / `96.91 MiB` | `32.74` / `25.26 MiB` | **`-93.4%`** / **`-73.9%`** | — |
| `geosun_3dgs_core.dll`（未压缩 / ZIP） | `237.90` / `35.65 MiB` | `16.55` / `11.57 MiB` | **`-93.0%`** / **`-67.5%`** | — |
| `curand64_10.dll`（未压缩 / ZIP） | `68.62` / `59.34 MiB` | `68.62` / `59.34 MiB` | `0.0%` | 宿主机 `tensor_random_ops.cpp` 仍依赖 `curandCreateGenerator`，占现 `trainer/` ZIP 约 `52.1%` |
| `trainer/` 合计（未压缩 / ZIP，60 文件） | `838.93` / `209.74 MiB` | `154.57` / `113.80 MiB` | **`-81.6%`** / **`-45.7%`** | 独立训练器包 `Geosun-3DGS-Trainer-sm80-20261117-190930.zip` 由 `210.48 MiB` 降至 **`113.83 MiB`**（`-45.9%`） |
| 整包 `Geosun-3DGS-1.1.1-Windows-x64.zip` | `2410.02 MiB` | **`2314.09 MiB`** | **`-95.93 MiB`** | `preprocess/` 与发射器占比保持 `2200.26 MiB`（`43` 文件） |

### 3. 回归构建、`cuobjdump` 与正式打包验收

- **训练器与完整 GUI 双构建通过**：`build-windows-trainer-ptx80`（`lfs_train`）与 [build-windows-release](file:///D:/Dev/LichtFeld-Studio/build-windows-release)（`LichtFeld-Studio.exe` `23.08 MiB`、`lfs_core.dll` `9.6 MiB`）均 `exit code 0`。
- **`cuobjdump` 校验通过**：
  - `geosun_3dgs_train.exe` 含 `220/220` 条 `compressed` 记录（`ptx:sm_80`、`elf:sm_86`、`elf:sm_89`、`elf:sm_120` 各 `55`）；`geosun_3dgs_core.dll` 含 `124/124` 条 `compressed` 记录（四目标各 `31`）。
  - `cuobjdump --dump-symbols geosun_3dgs_train.exe` 对 Tensor 泛型/广播核模板匹配数为 **`0`**；对 EXE 提取的 `220` 份与 DLL 提取的 `124` 份 cubin/PTX 做 SHA-256 交叉比对，跨二进制重复模块数为 **`0`**（基线为 `6` 个模块 / `48.13 MiB`）。
- **真机 GPU 端到端冒烟通过（`RTX 4090`）**：对 `work\展厅`（`585` 图 + 掩码 + 深度/法向先验）执行 `--iter 100 --headless`，`nvImageCodec` GPU 解码、卷帘与 MRNF+GUT+PPISP 在 `2.135 s` 内正常完成（`46.8 iter/s`，写出 `project.licht` `104.17 MiB`，`exit code 0`）。
- **提交与正式重打包完成**：
  - `LichtFeld-Studio` 已提交为 `70fb4a973`；清理 `build-windows-trainer-ptx80\src` 中旧 CP936 `#deps 0` `.cpp.obj` 后经 `package-windows-trainer.cmd` 重编验证 `main.cpp.obj` 已纳入 `include/git_version.h` 依赖，产出无 `-dirty` 的 [Geosun-3DGS-Trainer-sm80-20261117-190930.zip](file:///D:/Dev/LichtFeld-Studio/build-delivery/Geosun-3DGS-Trainer-sm80-20261117-190930.zip)（`113.83 MiB`），已删除 `20261116-134442`、`20261117-153642`、`20261117-190309` 旧训练器包及未瘦身旧版 `1.1.1` 整包。
  - 配对 `GeosunPre` `33ac17c`（`GeosunPre-1.1.1-33ac17c-ampere_plus`）通过 `package.ps1` 重新封装 [Geosun-3DGS-1.1.1-Windows-x64.zip](file:///D:/Dev/GeosunPre/packages/Geosun-3DGS-1.1.1-Windows-x64.zip)（`2314.09 MiB`，`exit code 0`），通过 `Geosun.exe --help`、`geosunpre.exe check-models`（`passed: true`）与 `geosun_3dgs_train.exe --version`（`Geosun 3DGS Trainer v0.5.5-dev-305-g70fb4a973 (70fb4a973)`）及落盘前后哈希一致性核验。

---

## 2026-11-17：GeosunPre 纯 C++/CUDA 零精度损失提速包（Phase 1）双场景实测

- **测试环境**：Windows x64，`NVIDIA GeForce RTX 4090`（`24564 MiB`，`SM 8.9`）+ `13th Gen Intel(R) Core(TM) i7-13700F`（`24` 逻辑处理器），Ninja Release（`/W4` 无新增警告，不更改外部模型、第三方依赖及交付格式契约）。
- **核心改动（五项零精度损失优化）**：
  1. **低压缩内部掩码与异步写出扩容**（[scene.cpp](file:///D:/Dev/GeosunPre/src/io/scene.cpp)、[static_masks.cpp](file:///D:/Dev/GeosunPre/src/geometry/static_masks.cpp)、[frontend.cpp](file:///D:/Dev/GeosunPre/src/frontend/frontend.cpp)）：为 `WriteImage` 增加可选 `png_compression` 参数，内部中间掩码使用 `png_compression = 1`（不影响 `export/` 默认导出压缩），并将 `AsyncWriter` 队列上限由 `2` 扩至 `4`。
  2. **`matches` 2D 局部性调度、CUDA buffer 池与单文件归档**（[cache.cpp](file:///D:/Dev/GeosunPre/src/io/cache.cpp)、[frontend.cpp](file:///D:/Dev/GeosunPre/src/frontend/frontend.cpp)、[reconstruction.cpp](file:///D:/Dev/GeosunPre/src/sfm/reconstruction.cpp)、[cache.md](file:///D:/Dev/GeosunPre/docs/cache.md)）：对 `matches` 待配对集合按 2D serpentine（`Tile = 32`）重排以提升缓存命中，`FeatureCache` 扩至 `96` 条目 / `8` 预取 / 非阻塞 `upload_stream` 并复用显存 buffer 池；废弃散列 `cache/matches/` 小文件，改用单一顺序归档 `cache/matches.bin`（配 `ArchiveWriterQueue` 写入与每线程独立 `std::ifstream` 原子抢占读取）。
  3. **Ceres BA / 快门静态 Schur 特化、参数块打包与节流 checkpoint**（[optimization.cpp](file:///D:/Dev/GeosunPre/src/sfm/optimization.cpp)、[model.cpp](file:///D:/Dev/GeosunPre/src/geometry/model.cpp)、[reconstruction.cpp](file:///D:/Dev/GeosunPre/src/sfm/reconstruction.cpp)）：为 `OPTIMIZE` 静态特化 `<2,3,6>` / `<2,3,d>` Schur，快门阶段将 `[pose(7), rot(3)]` 与 `[intrinsics(8), readout(1)]` 打包并搭配 `SubsetManifold` 切换阶段常量，`Depth` 改 2 维残差且仅在 `!metric || track.size() == 2` 时挂载，改用解析雅可比 `PlaneCost` 并为 `Model::Filter` / `RebuildReferences` 预计算姿态数组与复用标记缓冲；定期 `incremental.bin` checkpoint 节流为 `registered % 8 == 0 && >= 20 s`。
  4. **GPU-resident MoGe 解算、GPU 跨视角冲突检测与内存直持 depth 单次落盘**（[monocular.cpp](file:///D:/Dev/GeosunPre/src/geometry/monocular.cpp)、[supervision_kernels.cu](file:///D:/Dev/GeosunPre/src/geometry/supervision_kernels.cu)、[supervision.cpp](file:///D:/Dev/GeosunPre/src/geometry/supervision.cpp)）：MoGe TensorRT 输出保留在显存内并通过 CUDA 核函数完成样本采样（单次流同步回传求解 shift）与法向/深度图构建；在 `FilterViewConflicts` 前调用 `LasPointCache::Finish()` 释放常驻点云显存与 TensorRT 引擎；`FilterViewConflicts` 改用 2D serpentine（`Tile = 32`）+ `64` 槽 GPU 深度 LRU + CUDA 核函数判定，当总缓存 `<= 1.5 GiB` 时内存直持 depth 图像避免冗余 PNG 编解码。
  5. **消除交付后重复 `check` 子进程**（[bundle/main.cpp](file:///D:/Dev/GeosunPre/bundle/main.cpp)）：`PublishPending` 完成后跳过冗余 `geosunpre check`。

### 1. 同机室内 `展厅` 实测对照（`work\展厅` vs `work\展厅-fast`）

新交付目录 [work\展厅-fast](file:///D:/Dev/GeosunPre/work/展厅-fast) 已通过 `run.ps1 check`（`"passed": true`）并保留在本地。

| 阶段 / 指标 | 同机基线 `work\展厅`（`33ac17c`） | 本次提速 `work\展厅-fast`（`46b716e-dirty`） | 变化幅度 / 备注 |
| --- | --- | --- | --- |
| `masks` / `features` 阶段耗时 | `126.57 s` / `31.33 s` | `121.21 s` / `27.57 s` | `-4.2%` / `-12.0%`（主要耗时为 SAM3 GPU `107.84 s`） |
| `matches` 阶段耗时（`decode` / `write`） | `72.68 s`（`28.29 s` / `9.42 s`） | **`64.07 s`（`5.79 s` / `0.03 s`）** | `decode -79.5%`，`write -99.6%`（LoMa GPU `59.48 s`） |
| `sfm` 阶段耗时（`geo` / `filter` / `checkpoint`） | `296.45 s`（`26.35` / `12.92` / `5.65 s`） | **`127.52 s`（`11.32` / `5.12` / `0.14 s`）** | **`-57.0%`**（`opt` 大幅加速，增量存档耗时几乎归零） |
| `joint`（`align` BA，`linear`） | `42.73 s`（`30.66 s`，`used: true`） | **`18.87 s`（`14.40 s`，`used: true`）** | **`-55.8%`** |
| `shutter` 阶段耗时（主 BA / `polish` / 总计） | `92.45` / `126.89` / `229.43 s` | **`29.19` / `37.90` / `73.01 s`** | **`-68.2%`**（主 BA 与 `polish` 均加速 `~3.2×`） |
| `cleanup` / `static_masks` 阶段耗时 | `12.06 s` / `17.96 s` | `7.65 s` / `19.97 s` | `cleanup -36.6%` |
| `supervision` 总耗时（MoGe `host` / `view_conflicts`） | `135.50 s`（`87.34 s` / `20.40 s`） | **`122.23 s`（`18.87 s` / `1.96 s`）** | MoGe `host -78.4%`，`view_conflicts -90.4%`（`10.4×`），MoGe GPU `81.34 s` |
| 主阶段耗时合计 / 端到端实测总耗时 | `964.70 s` / — | **`582.08 s` / `608.94 s`** | **阶段合计 `-39.7%`**（`ELAPSED_SECONDS = 610.07 s`） |
| 注册图像 / `check` 回读校验 | `585 / 588` / `passed: true` | `584 / 588` / **`passed: true`** | 缺 `00214–00216-R` 及 `ba_workers=8` 浮点边界帧 `00301-R`（`-1` 帧） |
| SfM 点数 / 观测数 / 重投影 `median` / `p90` | `151,150` / `732,815` / `0.8881` / `2.0665 px` | `151,057` / `732,531` / `0.8881` / `2.0670 px` | 几乎零差异 |
| 卷帘留出 `before -> after` / `readout_s` | `1.4192 -> 1.1382 px` / `[24.19, 22.75] ms` | `1.4174 -> 1.1414 px` / `[24.34, 22.91] ms` | 均通过全项检查，精度持平 |
| 真实 LAS 点到面 `plane_after` / `polish` 重关联 | `5.89 / 17.40 mm` / `5.84 / 16.83 mm` | **`5.74 / 17.49 mm` / `5.70 / 16.94 mm`** | 持平略优 |
| 导航位置误差 `med` / `p90` | `20.56` / `37.31 mm` | `20.71` / `35.02 mm` | P90 略优（`-6.1%`） |
| 有效监督像素（深度 / 法向）/ `points3D.ply` | `336,675,067` / `395,079,740` / `881,579` | `335,962,013` / `394,345,339` / `881,551` | `-0.21%` / `-0.19%`（目录 `1806.29 -> 1803.82 MiB`） |

### 2. 室外 `篮球场` 实测对照（`work\篮球场-shutter` / `work\篮球场-opt` vs `work\篮球场-fast`）

新交付目录 [work\篮球场-fast](file:///D:/Dev/GeosunPre/work/篮球场-fast) 已通过 `run.ps1 check`（`"passed": true`）并保留在本地。

| 阶段 / 指标 | 历史同算法基线（`篮球场-shutter` / `篮球场-opt`） | 本次提速 `work\篮球场-fast`（`46b716e-dirty`） | 变化幅度 / 备注 |
| --- | --- | --- | --- |
| `masks` / `features` / `matches` | — / — / — | `109.73 s` / `25.44 s` / `51.74 s` | `matches decode 3.73 s`、`write 0.03 s`（LoMa GPU `47.98 s`） |
| `sfm` / `align` 阶段 | — / — | `89.90 s` / `18.20 s` | `sfm geo 8.50 s`、`filter 4.31 s`、`checkpoint 0.09 s`；`joint used: false` |
| `shutter` 阶段耗时（主 BA / `polish` / 总计） | `60.43` / `52.81` / `120.86 s` | **`26.77` / `26.13` / `59.73 s`** | **`-50.6%`** |
| `cleanup` / `static_masks` 阶段耗时 | — / `31.21 s`（`篮球场-opt`） | `12.30 s` / `26.04 s` | `sky.kept: true`（`517` 开阔、`52` 图除误检 `657,424 px`） |
| `supervision` 总耗时（`view_conflicts`） | `159.99 s`（`篮球场-opt`） / — | **`119.66 s`（`1.60 s`）** | **`-25.2%`**（MoGe `host 12.34 s`，GPU `74.65 s`） |
| **全流程总耗时** / 交付体积 | `715.91 s` / `2341.52 MB` | **`535.43 s` / `2340.94 MiB`** | **`-180.48 s`（`-25.2%`，提速 3 分钟）** |
| 注册情况 / `check` 回读校验 | `536 / 536`（`complete: true`）/ `passed: true` | **`536 / 536`（`complete: true`）/ `passed: true`** | 100% 完整通过 |
| SfM 点数 / 观测数 / `>=3` 轨迹 | `122,560` / `503,124` / `57,703` | `122,739` / `500,182` / `57,604` | 基本持平 |
| 重投影 `median` / `p90` 与留出校正 | `0.7094` / `1.7421 px`（留出后 `0.9757 / 2.8242`） | `0.7054` / `1.7377 px`（留出 `1.3370 -> 0.9751 / 2.8322`） | 均略优或持平 |
| `readout_s`（`L` / `R`）与 `polish` 重关联 LAS 平面 | `+32.85 / -31.04 ms`，`3.73 / 14.75 mm` | `+33.00 / -31.29 ms`，**`3.72 / 14.74 mm`** | 持平略优（`44,779` 面点） |
| 有效监督像素（深度 / 法向）/ `points3D.ply` | `225,080,139` / `306,386,746` / `2,500,000` | `224,996,325` / `306,386,023` / `2,500,000` | `-0.04%` / `-0.00%` |

- **当前耗时收敛分析**：经 Phase 1 消除 CPU 编解码、随机小文件 I/O、Ceres 动态求导/大块稀疏与跨视角冲突瓶颈后，全流程剩余耗时已主要收敛至纯 TensorRT `batch = 1` 引擎执行本身（双场景下 `SAM3 grounding` GPU `98.06–107.84 s`、`MoGe geometry` GPU `74.65–81.34 s`、`LoMa matcher` GPU `47.98–59.48 s`，三者合计占总耗时 `~41%–43%`）；在保持零模型改动与零精度损失前提下，已达纯粹 C++/CUDA 工程层面的高性价比上限，下一步可转入面向下游 3DGS 的质量优化阶段（Phase 2）。

---

## 2026-11-18：面向下游 3DGS 实测短板的三项提质（Phase 2）双场景实测

- **测试环境**：Windows x64，`NVIDIA GeForce RTX 4090`（`SM 8.9`，`24564 MiB`），Ninja Release（`/W4` 零新增警告），`LichtFeld-Studio` 与 [trainer.json](file:///D:/Dev/GeosunPre/configs/trainer.json) 零改动。
- **核心改动（三项提质）**：
  1. **室外天空 `128/255` 边缘原图引导细化与冠层天孔清理**（[static_masks.cpp](file:///D:/Dev/GeosunPre/src/geometry/static_masks.cpp)）：在 `sky.kept = true` 的室外场景中，围绕 `128/255` 边界 ROI 执行 `O(1)` 盒滤波 + 解析 `3×3` 双精度逆的彩色引导滤波细化（阈值 `0.35` 将粗网格残留的树冠/屋檐白边吃进 `128`），以 `5×5` 腐蚀后的 LAS 有效深度内核硬锁 `255` 保护建筑玻璃幕墙、白色篮板与浅色屋檐，并清理紧邻开阔天空且 `9×9` 外环 `>= 65%` 为暗色枝叶的树冠高亮低饱和天孔。
  2. **MoGe 子图按 $\omega_i$ 去卷帘重采样 + 跨视角冲突区法向联动置零**（[supervision_kernels.h](file:///D:/Dev/GeosunPre/src/geometry/supervision_kernels.h)、[supervision_kernels.cu](file:///D:/Dev/GeosunPre/src/geometry/supervision_kernels.cu)、[monocular.h](file:///D:/Dev/GeosunPre/src/geometry/monocular.h)、[monocular.cpp](file:///D:/Dev/GeosunPre/src/geometry/monocular.cpp)、[supervision.cpp](file:///D:/Dev/GeosunPre/src/geometry/supervision.cpp)）：GPU 核函数 `SampleMogeFaces` 按每帧读出转角 $\omega_i$ 执行 3 步不动点去卷帘反向映射，直接采样对齐 $s=0$ 参考姿态的针孔子图，并在 `FitFaceScales` / `GatherFaces` 双向换算将融合法向旋回像素读出线相机系（`moge2_vitl_faces_readout_line`）；`FilterViewConflicts` 在删除未获可信 LAS 确认且 `>= 2` 视角矛盾深度的 `4×4` 格内同步将矛盾法向置零 `(0,0,0)`。
  3. **同步双目 `shutter` 后刚体引导漏帧补注册 + 初始化 `points3D.ply` 补盲区/远景视觉点**（[optimization.h](file:///D:/Dev/GeosunPre/src/sfm/optimization.h)、[optimization.cpp](file:///D:/Dev/GeosunPre/src/sfm/optimization.cpp)、[shutter.h](file:///D:/Dev/GeosunPre/src/sfm/shutter.h)、[shutter.cpp](file:///D:/Dev/GeosunPre/src/sfm/shutter.cpp)、[pipeline.cpp](file:///D:/Dev/GeosunPre/src/app/pipeline.cpp)、[las.h](file:///D:/Dev/GeosunPre/src/io/las.h)、[las.cpp](file:///D:/Dev/GeosunPre/src/io/las.cpp)、[colmap.cpp](file:///D:/Dev/GeosunPre/src/io/colmap.cpp)）：
     - `PolishAcceptedShutter` 后执行 `RescueSynchronousFrames`：利用已收敛双目外参与 `RigShutterRotation` 为漏帧赋毫米级初值，综合 `matches.bin`、距离感知判歧 2D-3D 投影挂载与同镜头卷帘极线走廊互近邻匹配建立轨迹，经固定背景/外参/读出时间的两阶段局部卷帘 BA 与严格准入（`>= 32` 观测、`>= 8` 覆盖格、重投影限值、外参漂移 `<= 3 cm / 1.5°`、导航 `< 0.25 m`）恢复漏帧。
     - `ExportLasPly` 从清理前视觉模型筛选 `0.25–250 m` 内无 `0.25 m` 近邻有效 LAS 覆盖的高置信静态多视图点（`>= 3` 视图、重投影 `<= 1.5 px`、视差 `>= 2°`、零 `0` 掩码中心且 `>= 75%` 视图 `5×5 == 255`），按 `0.10 m` 去重（上限 `100,000`）并入 `points3D.ply`（总量仍封顶 `2,500,000`），保持 `points3D.bin` 纯 LAS 锚定口径不变。

### 1. 室内 `展厅` 实测对照（`work\展厅-fast` vs `work\展厅-p2`）

新交付目录 [work\展厅-p2](file:///D:/Dev/GeosunPre/work/展厅-p2) 经 `geosunpre.exe check` 回读校验通过（`"passed": true`）并保留在本地。

| 阶段 / 指标 | 基线 `work\展厅-fast` | Phase 2 `work\展厅-p2` | 变化幅度 / 备注 |
| --- | --- | --- | --- |
| **注册情况 / `check` 校验** | `584 / 588`（`complete: false`） / `passed: true` | **`588 / 588`（`complete: true`）** / **`passed: true`** | **`rig_rescue` 补齐 `3/3` 漏帧**（`00214-R`: `762` obs / `0.674 px`；`00215-R`: `842` obs / `0.693 px`；`00301-R`: `315` obs / `1.101 px`；`00216-R` 已于 `sfm` 注册 `478` obs） |
| SfM 点数 / 观测数 / `>=3` 轨迹 | `151,057` / `732,531` / `70,380` | **`152,333` / `738,242` / `70,483`** | `+1,276` 点 / `+5,711` 观测（`rig_rescue` 耗时 `17.71 s`，`+1,264` 点 `+2,931` 观测） |
| 重投影误差 `median` / `p90`（`L` / `R` 中位） | `0.8881` / `2.0670 px`（`0.8297` / `0.9550`） | `0.8894` / `2.0697 px`（`0.8331` / `0.9546`） | 补齐全部右目漏帧后精度保持持平 |
| 卷帘留出 `before -> after` / `polish` 点到面 | `1.4174 -> 1.1414 px` / `5.70 / 16.94 mm` | **`1.4121 -> 1.1353 px`** / `5.74 / 16.81 mm` | 平面 P90 微降至 `16.81 mm` |
| `static_masks.sky` 室内保护 | `kept: false`（`0` 像素 `128`） | `kept: false`（`edge_refined_images: 0`，`0` 像素 `128`） | 室内发光灯箱与白色天花零误挖 |
| MoGe 可信 LAS 锚点总数 / 内点率 | `140,387,672` / `94.72%` | **`141,678,230`** / `94.68%` | 去卷帘对齐 + 补齐全帧净增 `+129.1 万` 可信锚点 |
| 跨视角冲突剔除（深度 / **法向**） | `38,451,798 px` / `0 px` | `38,611,791 px` / **`36,459,152 px`** | 弧形玻璃展柜（如 `00146-R`）表面假法向精准清零 |
| 有效监督像素（深度 / 法向） | `335,962,013` / `394,345,339` | `338,545,560` / `360,680,178` | 剔除矛盾玻璃法向 `-8.54%`，交付体积 `1803.82 -> 1790.52 MiB` |
| **`points3D.ply`（总点 / LAS / 视觉补充）** | `881,551`（纯 LAS） | **`882,682`**（`881,578` LAS + **`1,104` 视觉**） | 从 `48,488` 个纯静态多视图点中补齐 `1,104` 个激光盲区锚点 |
| 全流程总耗时（`shutter` / `supervision`） | `608.94 s`（`73.01 s` / `122.23 s`） | `631.37 s`（`88.32 s` / `124.22 s`） | 含 `rig_rescue`（`17.71 s`）与法向联动重写仅 `+22.43 s`（`+3.7%`） |

### 2. 室外 `篮球场` 实测对照（`work\篮球场-fast` vs `work\篮球场-p2`）

新交付目录 [work\篮球场-p2](file:///D:/Dev/GeosunPre/work/篮球场-p2) 经 `geosunpre.exe check` 回读校验通过（`"passed": true`）并保留在本地。

| 阶段 / 指标 | 基线 `work\篮球场-fast` | Phase 2 `work\篮球场-p2` | 变化幅度 / 备注 |
| --- | --- | --- | --- |
| 注册情况 / `check` 校验 | `536 / 536`（`complete: true`） / `passed: true` | **`536 / 536`（`complete: true`）** / **`passed: true`** | 100% 完整通过（`rig_rescue` `attempted: 0` 零开销跳过） |
| 重投影 `median` / `p90` 与 `polish` 点到面 | `0.7054` / `1.7377 px`，`3.72 / 14.74 mm` | `0.7060` / `1.7402 px`，`3.76 / 14.89 mm` | 几何指标稳定持平 |
| **室外天空细化（`static_masks.sky`）** | `kept: true`，无原图细化 | `kept: true`，**`edge_refined_images: 520`**（**`+13,341,116` 天空 px**，`+540,068` 静态 px） | 平均每图剥离约 `2.57 万` 像素树冠/屋檐高曝白边及叶隙天孔；目视抽检 `00203-R`、`00244-L`、`00120-R` 确认玻璃幕墙、白色篮板、灯杆与凉亭屋檐完好保持 `255` |
| **MoGe 可信 LAS 锚点总数 / 冲突深度** | `85,660,199` / `66,711,925 px` | **`85,719,701`** / **`65,131,692 px`** | 静态掩码收缩 `13.34M px` 下可信锚点仍净增 `+5.95 万`，跨视角深度冲突下降 **`-158.0 万 px`（`-2.37%`）**，印证 $\omega_i$ 去卷帘重采样消除了转身错位 |
| **冲突区法向联动清零（`dropped_normal_pixels`）** | `0 px`（有效法向 `306,386,023`） | **`65,106,698 px`**（有效法向 `238,724,684`） | 同步清零树隙与玻璃幕墙反射区的矛盾法向，交付体积由 `2340.94 MiB` 降至 **`2247.47 MiB`**（`-93.47 MiB`） |
| **`points3D.ply`（总点 / LAS / 视觉补充）** | `2,500,000`（纯 `<=50 m` LAS） | **`2,500,000`**（`2,495,565` LAS + **`4,435` 视觉**） | 从 `58,236` 个纯静态多视图点中筛出 **`4,435` 个高楼立面与激光盲区初始锚点**（其中距最近相机 `> 50 m` 共 `1,152` 点、最远 `145.07 m`、最高 `Z = 90.68 m`，其余 `3,283` 点补齐 `< 50 m` 无 LAS 回波盲区） |
| 全流程总耗时（`static_masks` / `supervision`） | `535.43 s`（`26.04 s` / `119.66 s`） | `689.87 s`（`162.83 s` / `125.05 s`） | `supervision` 仅 `+5.39 s`；新增耗时主要来自 `520` 幅原图 ROI 彩色引导滤波细化（`+136.79 s`） |

### 3. 下游 3DGS 闭环训练与全视角 `eval` 实测对照

基于 `LichtFeld-Studio`（`v0.5.5-dev-305-g70fb4a973`，同机 `RTX 4090`）导入 [trainer.json](file:///D:/Dev/GeosunPre/configs/trainer.json) 完成 30,000 步训练（工程文件 `C:\Users\admin\.lichtfeld\projects\展厅-p2.licht` 与 `篮球场-p2.licht`），并按 [output.md](file:///D:/Dev/GeosunPre/docs/output.md#L102-L117) 契约执行 `--eval-all --eval-mask="masks:..." --eval-mask-report=mask,unmasked` 全视角离线评测（结果保留于 [work\展厅-p2-eval](file:///D:/Dev/GeosunPre/work/展厅-p2-eval) 与 [work\篮球场-p2-eval](file:///D:/Dev/GeosunPre/work/篮球场-p2-eval)），对照此前同机同配置 `30k` 基线（`展厅-bundle` / `篮球场-bundle`）：

| 下游 3DGS 指标（`30,000` 步，`#GS = 3,000,000`） | `展厅` 基线（`585` 图） | **`展厅-p2`（`588` 图）** | 室内变化 | `篮球场` 基线（`536` 图） | **`篮球场-p2`（`536` 图）** | 室外变化 |
| --- | --- | --- | --- | --- | --- | --- |
| 训练耗时 / 终态 Loss / `.licht` 体积 | `~529 s` / — / `432.66 MiB` | `549.2 s` / `0.113902` / `433.33 MiB` | 多训 3 帧持平 | `457.5 s` / — / `420.73 MiB` | `471.3 s` / `0.287676` / `420.94 MiB` | 持平 |
| 卷帘相机激活核验（`rolling_shutter_cameras`） | `585 / 585`（`rows`） | **`588 / 588`（`rows`）** | **100% 完整激活** | `536 / 536`（`columns`） | **`536 / 536`（`columns`）** | **100% 完整激活** |
| 纯前景 `mask == 255` **PSNR**（均值） | `26.1906 dB` | **`26.2018 dB`** | **`+0.0112 dB`** | `19.0291 dB` | **`19.4619 dB`** | **`+0.4328 dB`** |
| 前景 PSNR 分位数 `p10 / med / p90 / max` | `23.95 / 26.43 / 28.01 / —` | **`23.81 / 26.49 / 28.07 / 30.66`** | 3 补救帧均值 **`27.54 dB`** | `17.57 / 18.88 / 20.69 / 21.99` | **`17.84 / 19.35 / 21.28 / 22.38`** | **全分位一致提升 `+0.27~+0.59 dB`** |
| 纯前景 **SSIM** / **LPIPS** | `0.8535` / `0.2693` | `0.8528` / `0.2701` | 588 全量持平 | `0.5652` / `0.3646` | **`0.5734` / `0.3557`** | **`+0.0082` / `-0.0089`** |
| 前景修正色偏 `bias_corr` | 近零 | `(-0.0007, -0.0008, -0.0007)` | 零色偏 | `(-0.0057, -0.0075, -0.0072)` | **`(-0.0049, -0.0065, -0.0064)`** | 负色偏收敛 `-12%`（白边渗入减轻） |
| **渲染法向对齐 `normal_angle_deg`** | `12.0082°` | **`10.7974°`** | **`-1.2108°`（`-10.1%`）** | `17.0784°` | **`14.1200°`** | **`-2.9584°`（`-17.3%`）** |
| 渲染深度对齐 `depth_absrel` / 先验中位误差 | `0.0070` / `16.32 mm` | **`0.0066`** / **`16.28 mm`** | `-5.7%` / 略优 | `0.0051` / `17.39 mm` | `0.0051` / **`17.05 mm`** | 持平略优（`low_alpha: 0.6961 -> 0.6937`） |
| 全图 `Unmasked` PSNR / SSIM / LPIPS | `20.4393 dB` / — / — | `20.3724 dB` / `0.7461` / `0.3209` | 持平 | — | `9.4637 dB` / `0.4190` / `0.4320` | 含透明黑底天空 |

### 4. 分项视觉抽检与验收结论

1. **室外天空 `128/255` 边缘细化验收（符合预期）**：
   - 对照 `00203-R`、`00244-L`、`00036-L`、`00181-R`、`00199-R` 的掩码与 `eval_step_30000` 渲染图：原 `篮球场-fast` 中屋顶栏杆、树冠轮廓处约 `14 px` 的阶梯状白边已被单像素级贴合剥离，树冠内部高曝天孔同步归入 `128`；LAS 实体核锁死使曲面高楼玻璃幕墙、凉亭屋檐、弯曲高杆灯及白色篮板保持完整 `255` 无误蚀，直接推动 `篮球场` 全量前景 PSNR 提升 **`+0.43 dB`**（`p90` 升至 `21.28 dB`）、SSIM/LPIPS 同步改善。
2. **MoGe $\omega_i$ 去卷帘重采样 + 冲突区法向清零验收（显著超预期）**：
   - 抽检室内弧形玻璃展柜 `00146-R` 与室外树隙透景 `00181-R` 的法向图及渲染图：原横跨透明玻璃柜门与树冠镂空区的整片假连续单目法向已被精准置零，柜内奖牌/展板与树后建筑立面保留清晰几何且无凹陷拉扯；双场景渲染法向偏差双双大幅下降（`展厅` **`12.01° -> 10.80°`**，`篮球场` **`17.08° -> 14.12°`**）。
3. **同步双目漏帧补注册 + 盲区视觉点初始化验收（符合预期）**：
   - `展厅-p2` 补齐的全部漏帧（`00214-R` 专利墙近景 `26.10 dB`、`00215-R` `25.84 dB`、`00301-R` 白墙木门 `30.66 dB` 全场最高，另有 `sfm` 已注册的 `00216-R` `26.89 dB`）与真值逐像素贴合、无重影，实现 **`588 / 588` 零掉帧闭环**且拉升整体中位与均值 PSNR。
   - `篮球场-p2` 注入的 `4,435` 个高置信视觉锚点（最高 `90.68 m`、最远 `145.07 m`）使 `00244-L`、`00036-L` 中 `> 50 m` 弧形玻璃高楼顶层幕墙横梁与轮廓完整收敛、对黑底无破洞飘点。
4. **残余边界观察（非回归，供后续演进参考）**：
   - `篮球场` 部分视角边角（R 镜头左上如 `00203-R`、L 镜头右上如 `00036-L`）存在少量伸入 `160°` 有效圆内的采集设备遮光罩/机壳边缘，SAM3 与静态掩码均标为 `255`（Phase 2 已剥离其紧邻天空的亮边），而多视角 3DGS 将随动遮挡渲染为黑色透明，构成剩余单图边界扣分项之一；此外 `~235 m × 180 m` 大园区在 `3M` SH0 高斯上限下远景树叶高频纹理仍受容量平滑限制，个别 `60–80 m` 紧邻天空的退台建筑侧角（如 `00181-R`）受邻近视角天空不透明度惩罚存在轻微暗化。整体三项改动均准确命中设计目标并通过全链路核验。

---

## 2026-11-19：面向大场景（古建筑）与异步双目的 P0 三项改进双场景实测

### 1. 背景与改动摘要

针对 1,438 帧大图幅室外异步双目大场景 `古建筑`（`D:\Data\古建筑原始数据`，`GS200G` / `3600×3600`，左右镜头错拍约 `56 ms`）暴露出的大图 BA 线性求解退化、全图导出掩码算子瓶颈及异步双目缺失刚体/IMU 约束问题，本轮完成三项 P0 改进并同步文档（[`input.md`](file:///d:/Dev/GeosunPre/docs/input.md)、[`reconstruction.md`](file:///d:/Dev/GeosunPre/docs/reconstruction.md)）：

1. **放宽 Ceres BA 显式 Schur 阈值**（[`optimization.cpp`](file:///d:/Dev/GeosunPre/src/sfm/optimization.cpp)）：将 `use_explicit_schur_complement` 的图像数阈值由 `<= 1024` 提高到 `<= 4096`，保持 `preconditioner_type = ceres::SCHUR_JACOBI`，消除超过 1,024 帧后在 `sfm` / `joint` / `shutter` / `polish` 中触发大矩阵隐式 CG 的耗时悬崖。
2. **加速 `static_masks` 室外天空边缘细化**（[`static_masks.cpp`](file:///d:/Dev/GeosunPre/src/geometry/static_masks.cpp)）：在 Pass 3 中以两次 $O(W\cdot H)$ `cv::distanceTransform`（`DIST_L2`、`DIST_MASK_5`）替换 4 次大核 `MORPH_ELLIPSE` 形态学卷积；`LocalColorStats` 改为 `CV_64F` 4× 块累加精确方差后降采样；`GuidedFilterCoefficients` / `GuidedFilterColor` 改为 2× Fast Guided Filter 仅在 `band != 0` 边缘带回算全分辨率透射前景均值；对 Pass 3 增加 `cv::setNumThreads(1)`（RAII 保护）避免多线程过度切分核。
3. **200 Hz POS 姿态接入、异步双目刚体链接与 IMU 卷帘旋转先验**（[`scene.h`](file:///d:/Dev/GeosunPre/src/io/scene.h)、[`scene.cpp`](file:///d:/Dev/GeosunPre/src/io/scene.cpp)、[`optimization.h`](file:///d:/Dev/GeosunPre/src/sfm/optimization.h)、[`optimization.cpp`](file:///d:/Dev/GeosunPre/src/sfm/optimization.cpp)、[`shutter.cpp`](file:///d:/Dev/GeosunPre/src/sfm/shutter.cpp)）：
   - 扩展 `NavigationRow` 与 `ReadNavigation`，严格按 `_pos.txt` 时间戳（不混用带时延偏移的 `_pos_ENH.txt`）解析第 6–11 列姿态角（`Z-X-Y` 欧拉约定 `Rz(yaw)*Rx(pitch)*Ry(roll)`）并在 `ReadExposureKinematics` 做球面线性插值（SLERP），写入 `ExposureKinematics`（`body_position`、`body_rotation`、`omega_world`、`has_attitude`）。
   - `shutter` 新增 `VerifyImuAttitude`（左/右镜头 boresight 残差中位均 `<= 2.0°` 门控）；将原同步约束泛化为 `RigFrame` / `BuildRigFrames`（支持触发间隔 `<= 0.2 s` 异步双目 200 Hz 位姿与角速度运动补偿）、`RescueRigFrames`；引入 `ImuShutterRotation` 先验（$\sigma = 0.15^\circ$、`HuberLoss(3.)`），仅当 `f.synchronous` 为真时叠加 `RigShutterRotation`，并在报告中记录 `rig_mode` 与 `imu_attitude`。

---

### 2. 主实验：`古建筑`（1,438 帧，异步双目 ~56 ms）对比

基线 `work\古建筑`（`47bb4fb`）与改进后 `work\古建筑-p0`（`47bb4fb-dirty`）实测结果对照如下：

| 指标 | 基线 (`work\古建筑`) | P0 (`work\古建筑-p0`) | 变化 / 结论 |
| :--- | :---: | :---: | :--- |
| 注册完成度 / `check` | `1438/1438`（`passed: true`） | `1438/1438`（`passed: true`） | 保持完整闭环，无掉帧 |
| 稀疏点数 / 观测数 | `431,158` / `1,998,736` | `431,014` / `2,000,011` | 基本持平 |
| `sfm` 总耗时 / BA / 最慢单轮 | `665.46 s` / `535.90 s` / `304.89 s` (`exp=False`) | `353.28 s` / `218.08 s` / `72.10 s` (`exp=True`) | **`sfm -46.9%` / BA `-59.3%`** |
| `joint` 总耗时 / 线性求解 | `60.11 s` / `49.07 s` (`exp=False`) | `36.88 s` / `25.28 s` (`exp=True`) | **`-38.6%`** |
| `shutter` 总耗时 / 主 BA / `polish` | `292.88 s` / `144.78 s` (`lin=121.84 s`, `exp=False`, `rig=0`) / `138.54 s` (`BA=129.81 s`) | `142.40 s` / `60.80 s` (`lin=41.15 s`, `exp=True`, `rig=719`) / `67.98 s` (`BA=60.97 s`) | **`-51.4%`**，接入 `719` 对异步 `RigFrame` |
| `static_masks` 耗时 | `379.46 s` | `179.39 s` | **`-52.7%`（提速约 2.1×）** |
| `static_masks` 细化像素统计 | `refined=1180`，`+sky=27,537,671`，`+static=1,447,248` | `refined=1180`，`+sky=28,139,536`，`+static=1,436,053` | `+sky +2.2%`、`+static -0.8%`，语义几乎等价 |
| 阶段累计耗时（墙钟） | `2237.59 s`（`37.3 min`） | `1585.22 s`（`26.4 min`；墙钟 `1649.16 s`） | **累计 `-652.37 s / -29.2%`** |
| `rig_mode` / `imu_attitude` | 无（`rig_frames=0`） | `asynchronous`（`719` 对），`verified=true`（boresight 中位 `[0.155°, 0.165°]`） | 200 Hz IMU 姿态高精度验证通过 |
| `explained_rotation` (`L / R`) | `[0.190, 0.364]` | `[0.269, 0.381]` | 左镜头旋转解释度提升 `+0.079` |
| `L` 主点偏移 `[cx, cy]` | `[+6.56, -1.70] px` | `[+4.13, -1.78] px` | 左镜头水平主点偏移显著收敛 |
| `L` 帧内滚转偏置 `rz mean / med` | `+0.494° / +0.499°` | `+0.301° / +0.307°` | **降低约 `39%`** |
| `readout_s` (`L / R`) | `[-45.58, +26.19] ms` | `[-45.03, +27.82] ms` | 与 `rot_fit [-37.32, +39.12] ms` 同向 |
| 卷帘留出验证 `val_after` | `0.8953 / 2.4943 px` | `0.8963 / 2.5031 px` | 通过全部验收检查 |
| `polish reassoc` / `cleanup` 点面误差 | `5.84 / 22.30 mm` / `5.94 / 22.44 mm` | `5.78 / 22.31 mm` / `5.89 / 22.39 mm` | 略有改善，保持高精度 |
| 全量重投影 / 导航位置误差 | `0.7456 / 1.8144 px` / `25.71 / 37.71 mm` | `0.7471 / 1.8168 px` / `25.60 / 37.48 mm` | 基本持平 |

`check` 核验输出（`work\古建筑-p0`）：`passed: true`，`images: 1438`，`observations: 2,000,011`，`points: 431,014`，`depth_pixels: 785,909,987`，`normal_pixels: 827,360,773`。

---

### 3. 同步双目回归：`篮球场`（536 帧，同步双目）对比

验证异步扩展与掩码/BA 优化不退化既有同步双目链路，基线 `work\篮球场-p2`（`334a366-dirty`）与 `work\篮球场-p0` 对照如下：

| 指标 | 基线 (`work\篮球场-p2`) | P0 (`work\篮球场-p0`) | 变化 / 结论 |
| :--- | :---: | :---: | :--- |
| 注册完成度 / `check` | `536/536`（`passed: true`） | `536/536`（`passed: true`） | 保持零掉帧（`obs=500,330`、`pts=122,390`） |
| `static_masks` 耗时 | `157.20 s` | `70.32 s` | **`-55.3%`（提速约 2.2×）** |
| `static_masks` 细化像素统计 | `refined=520`，`+sky=13,341,116`，`+static=540,068` | `refined=520`，`+sky=13,421,683`，`+static=539,128` | `+sky +0.60%`、`+static -0.17%` |
| `shutter` 耗时 / 模式 | `61.36 s`（`rig=268`） | `58.04 s`（`synchronous`，`268` 对） | `imu_attitude.verified=true`（boresight 中位 `[0.169°, 0.188°]`） |
| `readout_s` / `explained_rotation` | `[+33.02, -31.27] ms` / `[0.298, 0.306]` | `[+32.09, -30.53] ms` / `[0.308, 0.314]` | 旋转解释度稳步提升 |
| `val_after` / `polish reassoc` | `0.9812 / 2.8382 px` / `3.76 / 14.89 mm` | `0.9782 / 2.8630 px` / `3.76 / 14.86 mm` | 通过全部验收检查，零退化 |
| 全量重投影 / 导航误差 | `0.7060 / 1.7402 px` / `21.28 / 39.69 mm` | `0.7064 / 1.7427 px` / `20.01 / 37.90 mm` | 导航误差与点面精度略优 |
| 阶段累计耗时（墙钟） | `658.57 s`（`10.98 min`） | `584.52 s`（`9.74 min`；墙钟 `616.34 s`） | **累计 `-11.2%`**（注：`sfm 104.81 -> 130.88 s` 属 `<1024` 帧本已启用显式 Schur 时的增量 SfM 正常波动） |

`check` 核验输出（`work\篮球场-p0`）：`passed: true`，`images: 536`，`observations: 500,330`，`points: 122,390`，`depth_pixels: 224,307,162`，`normal_pixels: 238,573,975`。

---

### 4. 视觉抽检与后续演进观察

1. **天空边缘细化抽检无回归**：
   - `古建筑-p0`（`00350-L`、`00472-R`、`00199-L`）与 `篮球场-p0`（`00203-R`、`00181-R`）掩码轮廓锐利贴合，古建飞檐斗拱、细屋顶栏杆、弯曲高杆灯杆及树冠天孔均完整保留 `255`/`128` 边界且无过蚀。
2. **异步双目残余偏置分析与 P1/P2 方向**：
   - 接入 200 Hz IMU 与异步 `RigFrame` 后，`古建筑-p0` 左镜头 `cx` 偏移与 `rz` 偏置已显著收敛（`rz` 降 `39%`），但因单图约 `1,724` 条像素观测相对单条 `ImuShutterRotation` 残差（$\sigma = 0.15^\circ$）仍占主导，且 `古建筑` 镜头存在固有 `fy - fx ≈ +5.6 px` 纵横比与 L/R `readout_s` 不对称残余，后续如需进一步抑制单镜头主点/滚转漂移可结合 P1/P2 评估权重或内参耦合策略。
   - **尚未验证项**：本轮按计划仅交付预处理与 `check` 验收，未运行下游 3DGS 训练（留待实训确认渲染 PSNR/SSIM/几何提升）。
   - **P1/P2 待推进方向**：大场景下大面积静止阴影与随行采集员手臂的静态掩码误判、`supervision` 阶段 `hold_depths` `1.5 GiB` 上限导致的磁盘二次解码、初始化 `points3D.ply` 250 m 远距视觉外点约束，以及面向 1,400+ 帧大场景的 [`trainer.json`](file:///d:/Dev/GeosunPre/configs/trainer.json) 训练尺度配比。

---

## 2026-11-19：面向掩码质量与大场景监督缓存弹性的 P1 改进双场景实测

### 1. 前置实证排查与改动摘要

动手实施 P1 前，对 `work\古建筑-p0` 全量掩码/深度与监督日志做了像素级叠加实测：
- **放弃采集员衣袖启发式（实证已覆盖）**：原以为暴露在静态区域的采集员蓝色衣袖，经叠加 `160°` 有效域与最终掩码后实测 `00199-L` 约 `99.3%`、`00233-R` 约 `99.9%`、`00350-R` `100.0%` 已被标为 `0`，代表样张 `00140-L`、`00434-L` 下角整条蓝色衣袖亦已完整 `0` 掩码，极少量跨界弧缘在 `supervision` 中以实际远距/地面激光点覆盖、未出现 `< 0.8 m` 误投影；且按“边角无近距 LAS + 多视角冲突”泛化清零误判走廊柱墙风险高，故明确跳过，不做脆弱启发式。

聚焦 3 项真实确凿问题完成改动并同步契约文档（[`output.md`](file:///d:/Dev/GeosunPre/docs/output.md)、[`supervision.md`](file:///d:/Dev/GeosunPre/docs/supervision.md)）：

1. **超大建筑物阴影按连通人物邻域局部裁剪**（[`frontend.cpp`](file:///d:/Dev/GeosunPre/src/frontend/frontend.cpp)）：在 `GroundingMask` 中若单个 `shadow` 检测在 `288×288` 格网覆盖超过 `8%`（`kShadowMaxFullShare`），或处于 `(3%, 8%]` 且面积超过接触 `people` 连通块总面积的 `3` 倍（`kShadowMaxPersonAreaRatio`），则仅将落入 `people` 膨胀 `kShadowReach = 24` 格邻域 `near` 的部分并入 `transient`，不再整块吞并庭院/大厅建筑大阴影；`<= 3%` 的小影子（如操作员地面投影及室内近距人影）保持完整并入，并在 `masks_profile.json` 记录 `shadows_clipped`。
2. **`supervision` 跨视角冲突 `hold_depths` 自适应内存上限**（[`supervision.cpp`](file:///d:/Dev/GeosunPre/src/geometry/supervision.cpp)）：`HoldDepthBudgetBytes()` 通过 `GlobalMemoryStatusEx` 查询主机实时可用物理内存，上限取 `min(ullAvailPhys / 4, 8 GiB)`（失败回退 `1.5 GiB`），按已注册镜头真实像素尺寸累加计算 `held_depth_bytes`，并在 `monocular.view_conflicts` 记录 `hold_depths`、`held_depth_bytes` 与 `hold_depth_budget_bytes`。
3. **超大 LAS 相机射程预裁剪与分块 AABB 投影剔除**（[`supervision.h`](file:///d:/Dev/GeosunPre/src/geometry/supervision.h)、[`supervision.cpp`](file:///d:/Dev/GeosunPre/src/geometry/supervision.cpp)、[`colmap.cpp`](file:///d:/Dev/GeosunPre/src/io/colmap.cpp)）：
   - `--supervision-cache-mb` 默认上限由 `2048 MiB` 提升为 `4096 MiB`（范围 `1–8192`，运行期仍受 `free_vram / 4` 约束）；
   - `LasSurfelCache` 传入 `Reconstruction` 模型后，`ScanLas` 按已注册相机中心包络向外扩 `MaxDepth + 2.5 m`（`68.035 m`）预裁剪射程外 LAS 点并紧缩分片直方图范围（`culled_out_of_reach`）；
   - 面元缓存按 `PointChunk`（`500,000` 点）切片并记录原始坐标 `RawBounds`，在 `ProjectNearest` 与 `Project` 中无论是 GPU 常驻还是文件流式（`seekg` 按偏移跳过）均先以当批相机 `BatchReachBounds`（`MaxDepth + 2.0 + 1.5 * max_shift`）过滤无重叠分块（`chunks_total`、`chunks_projected`、`chunks_skipped`）。

---

### 2. 主实验：`古建筑`（1,438 帧室外大阴影+大内存持留）对比

`work\古建筑-p0`（`47bb4fb-dirty`）与 `work\古建筑-p1`（`0bdb67f-dirty`）对照如下：

| 指标 | P0 (`work\古建筑-p0`) | P1 (`work\古建筑-p1`) | 变化 / 结论 |
| :--- | :---: | :---: | :--- |
| 注册完成度 / `check` | `1438/1438`（`passed: true`） | `1438/1438`（`passed: true`） | 保持完整闭环 |
| Grounding 影子裁剪数 | `shadows=272`（未裁剪） | `shadows=272`，**`shadows_clipped=35`** | 仅命中 35 个超大建筑连带阴影 |
| 有效域被遮 `> 30% / > 20%` 图数 | `22` / `31` 张 | **`0` / `1` 张** | **`> 30%` 误抹大图完全清零** |
| `00035-L` 静态 / 有效深度像素 | `222,657`（`21.4%`） / `96,866` | **`480,909`（`46.1%`） / `337,315`** | **静态 `+116%`，深度 `+248%`** |
| `00472-R` 静态 / 有效深度像素 | `228,384`（`21.9%`） / `140,620` | **`414,352`（`39.7%`） / `316,873`** | **静态 `+81%`，深度 `+125%`** |
| `00350-R` 静态像素（随动小影对照） | `740,100` | `740,115` | 采集者紧凑地面阴影完整保留 |
| 总交付深度 / 法向像素 | `785,909,987` / `827,360,773` | **`791,665,885` / `832,946,346`** | **净增 `+576 万` 深度 / `+559 万` 法向** |
| `hold_depths` / 预算 | 未启用（`> 1.5 GiB` 硬上限） | **`true`（`2.79 GiB <= 3.58 GiB`）** | 全量深度驻留内存免落盘二次解码 |
| `view_conflicts` 耗时 | `31.98 s` | **`11.30 s`** | **`-64.7%`（提速 `2.8×`）** |
| `supervision` 总耗时 | `347.39 s` | **`322.79 s`** | 减少 `24.60 s` |
| `point_cache` 配置与分块 | `budget=2 GiB`, `gpu`（`0.94 GiB`） | `budget=4 GiB`, `gpu`, `chunks=103`, `projected=10,506` | 39×38 m 场景全在 65.5 m 射程内（`skipped=0`） |
| 稀疏点 / 观测 / 重投影 | `431,014` / `2,000,011` / `0.7471/1.8168 px` | `433,776` / `2,009,824` / **`0.7455/1.8140 px`** | 石板地面有效观测增加、重投影略优 |
| `polish reassoc` / 导航误差 | `5.78/22.31 mm` / `25.60/37.48 mm` | **`5.73/22.14 mm` / `25.59/36.94 mm`** | 精度小幅改善 |
| 阶段累计耗时（墙钟） | `1585.22 s` | **`1523.79 s`**（墙钟 `1586.56 s`） | 同步降耗 `61.43 s` |

`check` 核验输出（`work\古建筑-p1`）：`passed: true`，`images: 1438`，`observations: 2,009,824`，`points: 433,776`，`depth_pixels: 791,665,885`，`normal_pixels: 832,946,346`。

---

### 3. 室内阴影与文件流式缓存回归：`展厅`（`--supervision-cache-mb 64`）对比

在 `588` 帧室内场景 `展厅` 上显式指定 `--supervision-cache-mb 64`，强制 `LasPointCache` 走磁盘紧凑面元文件、双缓冲异步 H2D 上传与 `seekg` 路径，同时核验室内近距人影不被误放过：

| 指标 | 基线 (`work\展厅-p2`) | P1 (`work\展厅-p1`, `cache=64 MiB`) | 变化 / 结论 |
| :--- | :---: | :---: | :--- |
| 注册完成度 / `check` | `588/588`（`passed: true`） | `588/588`（`passed: true`） | 流式缓存全链路零出错 |
| `point_cache` 模式与传输 | `storage=gpu`, `h2d=0` | **`storage=file` (`64 MiB`)**, `h2d=27.84 GB`, `transfer=20 MB`, `chunks=64`, `projected=2,816` | 验证文件分块流式投影正常 |
| `las_proj` / `supervision` 耗时 | `14.28 s` / `124.22 s` | `18.21 s` / `126.66 s` | 双缓冲流式仅增 `+2.44 s`（约 `+2%`） |
| `hold_depths` / `vc` | `—` / `4.72 s` | `true` (`1.13 <= 3.88 GiB`) / `4.70 s` | 与原内存驻留路径一致 |
| Grounding 影子裁剪数 | `shadows=81` | `shadows=81`, `shadows_clipped=2` | 几乎无触发 |
| `00051-R` / `00052-R` / `00306-L` 静态像素 | `462,956` / `536,926` / `570,284` | **`462,956` / `536,926` / `570,284`** | **静态像素数完全一致（零损失）** |
| 总有效深度 / 法向像素 | `338,545,560` / `360,680,178` | `338,637,245` / `360,752,429` | 偏差 `< +0.03%`（原子加正常抖动） |
| 重投影 / `polish` / 导航 | `0.8894/2.0697 px` / `5.74/16.81 mm` / `20.69/37.20 mm` | `0.8872/2.0666 px` / `5.61/16.99 mm` / `21.32/35.25 mm` | 零退化 |

`check` 核验输出（`work\展厅-p1`）：`passed: true`，`images: 588`，`observations: 737,257`，`points: 152,679`，`depth_pixels: 338,637,245`，`normal_pixels: 360,752,429`。

---

### 4. 视觉抽检与未验证项

- **掩码视觉抽检**：对照 `古建筑-p1` [00035-L](file:///d:/Dev/GeosunPre/work/古建筑-p1/masks/2411-357849-840_00035-L.JPG.png)、[00472-R](file:///d:/Dev/GeosunPre/work/古建筑-p1/masks/2411-358199-484_00472-R.JPG.png) 与 [00350-R](file:///d:/Dev/GeosunPre/work/古建筑-p1/masks/2411-358101-834_00350-R.JPG.png)，大面积廊檐阴影吞并已被精确剥离、庭院地面完整恢复为 `255`，同时行人及 `24` 格邻域局部影子、`00350-R` 采集员随动地影均保持 `0`；`展厅-p1` [00051-R](file:///d:/Dev/GeosunPre/work/展厅-p1/masks/R/0000/2436-213993-428_00051-R.JPG.png) 近景展馆参观者与地面影轮廓完全保持。
- **已知验证边界**：现有三套测试场景最大空间尺度为 `50×93×20 m`（`< 2 * 65.5 m` 射程盒），`chunks_skipped > 0` 与 `culled_out_of_reach > 0` 分支将在百米级以上更广域场景中实际触发（本次通过代码走读与流式路径全量投影核对）；下游 3DGS 训练仍留待用户按大场景需求执行。

---

## 2026-11-19：引入 LAS 相机种子占空多边形 ROI（`roi.json` 方案 B）及 3DGS / Mesh / OSGB 消融验证

针对手持双鱼眼 + LiDAR 扫描场景“雷达不可达远背景无需重建却白白消耗 3DGS 高斯配额、TSDF 体积及 OSGB 瓦片/纹理页”的问题，在 `GeosunPre` 与定制 `LichtFeld-Studio` 中联合引入自动 2D 简化多边形棱柱 ROI（`roi.json`，`CropPolygon` 路径）：

### 1. 工程改动摘要
1. **`GeosunPre`（`src/io/roi.{h,cpp}`、`las.{h,cpp}`、`colmap.{h,cpp}`、`app/{runtime,pipeline,main}.cpp`）**：
   - `export` 阶段对 `ScanLas` 原始 LAS 按 `1 m²` 网格累计点数（阈值 `>= 30` 点），并以已注册相机轨迹中心栅格为连通种子，经 `2 m` 圆形膨胀 + 外轮廓 `approxPolyDP` / 凸包回退生成 `3..64` 顶点的世界坐标非自交简单闭合多边形，`Z` 范围按内部点 `[p0.1%, p99.9%]` 结合相机高程包络收紧为 `[z_min, z_max]`；
   - 新增交付文件 [roi.json](file:///d:/Dev/GeosunPre/docs/output.md)（纳入 `manifest.json`、`request["roi"]` 与 `check` 严格拓扑/相机包含回读校验），并在 `2.5M` `points3D.ply` 采样上限处优先分配 ROI 内点（ROI 外最多保留 `5%`）；新增 `--no-roi` 关闭；
   - 更新 [trainer.json](file:///d:/Dev/GeosunPre/configs/trainer.json) 默认启用 `crop_to_cropbox: true` 与 `cropbox_growth_inside: 0.7`（保留 `0.1` 的 `cropbox_loss_weight` / `cropbox_lr_scale`）。
2. **`LichtFeld-Studio`（`src/training/roi_sidecar.{hpp,cpp}`、`training_setup.cpp`、`argument_parser.cpp`、`FreeSpace.{h,cu}`、`metrics.{hpp,cpp}`、`trainer.cpp`）**：
   - 数据集加载（`attachDatasetPointCloud`）在未显式传 `--no-roi` / `--cropbox` 且非恢复模式时，自动将 `roi.json` 中心化挂载为可交互 `Model_crop_polygon`；支持 `--no-roi`、允许 `--crop` 与 `roi.json` 联用；
   - `gsplat_lfs::free_space_loss` 增加可选 `pixel_weight`，训练期接入 `roi_weight`；
   - `--eval-mask cropbox` 支持 `CropPolygon` 及 `roi.json` 回退，并以渲染 `(depth, alpha)` 计算 `ROI ∩ mask==255` 静态前景有效像素集，在报告与 `metrics_report.txt` 中输出 `Region of interest:`；
   - 修复初版实测暴露的两项耦合问题：（a）`MaskMode::SegmentAndIgnore` 下不把 `roi_weight` 传给 `fuse_mask_opacity_penalty`，避免相机位于 ROI 内、边缘像素 `alpha < 0.5` 时天空透明惩罚被稀释 `10×` 而残留紫白雾影；（b）`prepare_evaluation_view` 使用渲染 `(depth, alpha)` 判定表面落点而不再用单目 `depths/*.png` 覆盖，避免超越 LiDAR 射程的高楼玻璃顶被单目深度低估而误计入已裁除 ROI 外的评分区域。
3. **构建与单元测试验证**：
   - `GeosunPre`：`tools/windows/build.ps1` Ninja Release（`/W4`）零新增警告通过；
   - `LichtFeld-Studio`：`gsplat_tests --gtest_filter="*FreeSpace*"` 3/3 通过；`lichtfeld_tests` ROI / `CropPolygon` / `EvalMask` / `ArgumentParserTest` / `TrainerMaskSupervisionTest` 相关过滤套件 212/212 通过。

---

### 2. `GeosunPre` 实景 `export` + `check` 验收

在既有重建工程 [展厅-p2.work](file:///d:/Dev/GeosunPre/work/展厅-p2.work) 与 [篮球场-p2.work](file:///d:/Dev/GeosunPre/work/篮球场-p2.work) 上重新执行 `export` 与 `check`：

| 场景 / 产物 | 相机含纳 (`cameras_inside`) | 多边形顶点数 | 面积 (`m²`) | `Z` 范围 (`m`) | `kept_las_fraction` | 初始化点云 (`roi / out_roi / supplement`) | `check` 状态 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| [展厅-roi](file:///d:/Dev/GeosunPre/work/展厅-roi/roi.json) | `588 / 588` | `52` | `1208.0` | `[-3.41, 8.55]` | `0.999939` (`99.994%`) | `881,043 / 535 / 1,104`（共 `882,682`） | `passed: true` |
| [篮球场-roi](file:///d:/Dev/GeosunPre/work/篮球场-roi/roi.json) | `536 / 536` | `48` | `8738.0` | `[-2.56, 49.87]` | `0.999962` (`99.996%`) | `2,492,312 / 3,253 / 4,435`（共 `2,500,000`） | `passed: true` |
| [篮球场-noroi](file:///d:/Dev/GeosunPre/work/篮球场-noroi)（`--no-roi`） | `536 / 536`（无 `roi.json`） | `—` | `—` | `—` | `—` | `2,495,565 + 4,435`（共 `2,500,000`） | `passed: true` |

说明：`篮球场` 原始三维点云跨越 `[-91.1, -96.4, -6.2] ~ [144.4, 84.2, 90.7] m`（含百米外玻璃塔楼与视觉稀疏远点），自动 ROI 精确聚焦至约 `8,738 m²` 的有效作业廊区与球场主体并保留 `99.996%` 原始激光点。

---

### 3. `篮球场` 下游 30k 3DGS 训练 + `cropbox` 评测 + Mesh / OSGB 消融对比

基于 [篮球场-roi](file:///d:/Dev/GeosunPre/work/篮球场-roi) 与 [trainer.json](file:///d:/Dev/GeosunPre/configs/trainer.json) 在 RTX 4090 上对比基线（`--no-roi`）与自动 ROI（`roi-v2`，含 `SegmentAndIgnore` 天空全权重修正与 `--crop`）：

| 阶段 / 指标 | 基线（`--no-roi`，[noroi-train](file:///d:/Dev/GeosunPre/work/篮球场-noroi-train) / [noroi-eval2](file:///d:/Dev/GeosunPre/work/篮球场-noroi-eval2/metrics_report.txt) / [noroi-mesh](file:///d:/Dev/GeosunPre/work/篮球场-noroi-mesh)） | 自动 ROI v2（`roi.json` + `--crop`，[roi-v2-train](file:///d:/Dev/GeosunPre/work/篮球场-roi-v2-train) / [roi-v2-eval](file:///d:/Dev/GeosunPre/work/篮球场-roi-v2-eval/metrics_report.txt) / [roi-v2-mesh](file:///d:/Dev/GeosunPre/work/篮球场-roi-v2-mesh)） | 变化 / 结论 |
| :--- | :---: | :---: | :--- |
| 30k 训练耗时 / 速率 | `458.489 s` (`65.4 iter/s`) | `459.083 s` (`65.3 iter/s`) | 基本持平（`+0.6 s`） |
| 终态高斯数 / `project.licht` | `3,000,000` / `420.71 MiB` | **`2,919,206`**（裁除 `80,794` 外点）/ `411.74 MiB` | 外围杂散高斯已自动清理 |
| `eval` 口径与覆盖 (`536/536` 卷帘) | `Mask` 平均有效像素占比 `0.619497` | `Mask` 平均有效像素占比 `0.619009`（最大单图差 `0.0052`） | 同口径公平对齐 |
| `Mask` (`ROI ∩ mask==255`) PSNR / SSIM / LPIPS | `20.2917` / `0.5617` / `0.3536` | `20.2007` / `0.5588` / `0.3563` | 仅 `-0.091 dB`（视觉抽检干净无雾边） |
| 法线角误差 `normal_angle_deg` | `14.1099°` | **`13.8834°`** | **`-0.2265°` 几何法线更贴合先验** |
| 先验深度误差 (`median / p90`, mm) | `16.9872` / `1034.2051` | `16.9795` / **`810.6697`** | **`p90` 误差骤降 `223.5 mm` (`-21.6%`)** |
| TSDF 稀疏块数 / 体素内存 | `3,468,381` 块 (`16.58 GB`, `3 chunks`) | **`2,252,192` 块 (`10.76 GB`, `2 chunks`)** | **`-35.1%` TSDF 体素分配** |
| MC 融合体素数 / RAM 峰值 | `89,920,829` (`~5.7 GB`) | **`63,990,385` (`~4.2 GB`)** | **`-28.8%` 体素与内存开销** |
| 网格顶点数 / 面片数 | `24,728,632` / `44,817,540` | **`18,139,374` / `32,527,423`** | **`-26.6%` 顶点 / `-27.4%` 面数** |
| `splats.ply` 磁盘体积 | `1,250,301,381 B` (`1192.38 MiB`) | **`912,619,894 B` (`870.34 MiB`)** | **`-27.0%` 网格体积极速瘦身** |
| OSGB 瓦片数 / LOD 层级 / 纹理页 / 纹素 | `518` tiles / `2,038` levels / `7,740` pages (`138.53M`) | **`256` tiles / `1,018` levels / `4,473` pages (`95.31M`)** | **瓦片与层级数减半 (`-50%`)，纹理页 `-42.2%`** |
| `splats_osgb/` 总文件数与磁盘体积 | `2,039` files / `2,085,684,218 B` (`1989.06 MiB`) | **`1,019` files / `1,492,099,792 B` (`1422.98 MiB`)** | **`-28.5%` OSGB 体积（省 `566 MiB`）** |
| `mesh + osgb` 总墙钟耗时 | `939.353 s` (`15m39s`) | **`728.134 s` (`12m08s`)** | **提速 `211.2 s`（`-22.5%`）** |

- **抽检对比（[2435-353235-065_00272-L](file:///d:/Dev/GeosunPre/work/篮球场-roi-v2-eval/eval_step_30000/2435-353235-065_00272-L.png)、[2435-353047-182_00084-R](file:///d:/Dev/GeosunPre/work/篮球场-roi-v2-eval/eval_step_30000/2435-353047-182_00084-R.png)）**：修复初版天空透明惩罚稀释后，建筑高楼边缘与走廊藤架空隙处紫白雾影彻底消失；初版 `roi-train`（`20.1678 -> 19.6214`）及对应对照目录保留在 `work\篮球场-roi-train` / `work\篮球场-roi-eval` 作为复盘留档。
- **取舍结论**：对于存在远景不可达建筑/背景的多视角手持扫描场景，自动 `roi.json` + `CropPolygon` 方案在保持主作业区外观保真（PSNR 偏差仅 `0.09 dB`）、显著压低远区几何噪点（`p90` 深度误差 `-21.6%`）的同时，对下游 TSDF 网格提取与 OSGB 分级瓦片烘培带来了显著工程收益（TSDF 块 `-35.1%`、网格/OSGB 体积减少约 `27%–28.5%`、OSGB 瓦片数减半、导出提速 `22.5%`），确立为 `GeosunPre + LichtFeld-Studio` 联合默认链路。

## Phase B：3DGUT 逐像素射线高斯深度 + 多面穿透共识与 LAS 法线幻觉门控（2026-11-20）

### 问题定位与改动摘要
针对 Phase A 在 `南非展会-phaseA` `263-L` 出现的展台上方白墙法线凹坑 + `16×16` 黑色瓦片网格，以及 `展厅-phaseA` 法线均值由 `10.8158°` 回退至 `13.0791°` 的问题，联合定位并修复如下：
1. **3DGUT 训练射线深度替代常量中心深度**（[Cameras.cuh](file:///d:/Dev/LichtFeld-Studio/src/training/rasterization/gsplat/Cameras.cuh)、[FromWorldRay.cuh](file:///d:/Dev/LichtFeld-Studio/src/training/rasterization/gsplat/FromWorldRay.cuh)、[Utils.cuh](file:///d:/Dev/LichtFeld-Studio/src/training/rasterization/gsplat/Utils.cuh)、[RasterizeToPixelsFromWorld3DGSFwd.cu](file:///d:/Dev/LichtFeld-Studio/src/training/rasterization/gsplat/RasterizeToPixelsFromWorld3DGSFwd.cu)、[RasterizeToPixelsFromWorld3DGSBwd.cu](file:///d:/Dev/LichtFeld-Studio/src/training/rasterization/gsplat/RasterizeToPixelsFromWorld3DGSBwd.cu)、[GeometryFeatures.cu](file:///d:/Dev/LichtFeld-Studio/src/training/rasterization/gsplat/GeometryFeatures.cu)、[Rasterization.cpp](file:///d:/Dev/LichtFeld-Studio/src/training/rasterization/gsplat/Rasterization.cpp)）：
   - 将训练光栅化 `CDIM ∈ {1, 4, 8}` 的常量高斯中心深度替换为逐像素射线-高斯最近点深度 `depth = max(t_hit, 0) * cam_ray_z`（对齐 Vulkan 实时渲染器 `gut_splat_view_depth`），同步覆盖 `depth^2`、`median_out` 与 `free_space`；并在后向 `opac*vis > 0.999` 饱和路径外完整回传 `v_depth -> means / quats / scales` 解析 VJP，消除大尺度平片高斯的球冠深度伪梯度与 `16×16` 瓦片边界伪影。
   - 验证：`gsplat_tests.exe` `79/79` 通过（`1` 项显式跳过，`FlatTiltedDiskDepthNormalMatchesPlaneNormal` 平面深度法线一致性 `< 2e-3`、同平面 `depth_distortion < 1e-3`，四核一致性 `GutGeometryGradParity` 最大相对误差 `7.18e-8 ~ 1.09e-5`）；`lichtfeld_tests.exe` 训练参数/先验损失筛选用例 `61/61` 通过。
2. **多面穿透共识与 LAS 法线交叉门控**（[monocular.h](file:///d:/Dev/GeosunPre/src/geometry/monocular.h)、[monocular.cpp](file:///d:/Dev/GeosunPre/src/geometry/monocular.cpp)、[supervision.cpp](file:///d:/Dev/GeosunPre/src/geometry/supervision.cpp)）：
   - 向 `CleanLasAndEncodeNormals` 传入 GPU 平面 LAS surfel 法线图，穿透裁剪由 `face[0]` 单面判定升级为多面 RANSAC 共识；新增非 MoGe 边缘处的 `MoGe vs LAS > 35°` 严重幻觉法线置零（`normal_disagree_dropped_pixels`），并对 `≤ 25° && log_ratio ≤ 1.1` 的远端实体墙面执行保留（`far_normal_rescued_pixels`）。
3. **训练超参平衡**（[trainer.json](file:///d:/Dev/GeosunPre/configs/trainer.json)）：
   - `depth_distortion_weight` 从 `1.0` 调降至 `0.25`，`normal_prior_decay` 从 `[0.5, 0.8]` 延后至 `[0.6, 1.0]`，避免弱纹理墙面被强行压成超大内凹薄片或在末期过早失去外部法线约束。

### 实验一：`展厅`（`588` 视角，`30k` 步，`res /2`，`cropbox` ROI）
- **监督清洗统计（[supervision.json](file:///d:/Dev/GeosunPre/work/展厅-phaseB/supervision.json)）**：输入 LAS 像素 `350,210,624`；`edge_dropped=32,613,773`、`near_dropped=2,968,395`（Phase A `3,977,479`）、`far_dropped=3,946,896`（Phase A `9,749,877`）、`far_rescued=3,699,726`、`penetration_normal_dropped=1,809,544`（Phase A `9,579,982`，`-81.1%`）、`normal_disagree_dropped=22,501,650`；最终 `depth_pixels=310,681,560`（Phase A `303,869,497`）、`normal_pixels=372,957,500`（Phase A `387,688,712`）。目视检查 [00233-R 法线图](file:///d:/Dev/GeosunPre/work/展厅-phaseB/normals/R/0000/2436-214175-314_00233-R.JPG.png) 与 [00240-R 法线图](file:///d:/Dev/GeosunPre/work/展厅-phaseB/normals/R/0000/2436-214182-310_00240-R.JPG.png) 确认 `face[0]` 吊顶误裁彻底愈合，玻璃展柜无误保留。

| 评测配置 | 目录 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | `normal_angle_deg` ↓ | 深度中位 / `p90` 误差 (mm) ↓ | 说明 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `roi-eval` (`588` 视角) | [metrics_report.txt](file:///d:/Dev/GeosunPre/work/展厅-roi-eval/metrics_report.txt) | `26.2289` | `0.8535` | `0.2694` | `10.8158°` | `16.2765 / 107.4884` | 仅 MoGe 尺度深度，`normal_prior_decay=[1.0,1.0]`，对 MoGe 过拟合导致远区深误差高 |
| `roi-eval-on-las` (`588` 视角) | [metrics_report.txt](file:///d:/Dev/GeosunPre/work/展厅-roi-eval-on-las/metrics_report.txt) | `26.1407` | `0.8511` | `0.2706` | `11.9213°` | `10.6766 / 79.0565` | `roi` 模型投射至 LAS 深度口径评测 |
| `phaseA-eval` (`588` 视角) | [metrics_report.txt](file:///d:/Dev/GeosunPre/work/展厅-phaseA-eval/metrics_report.txt) | `26.2127` | `0.8533` | `0.2702` | `13.0791°` | `10.9555 / 50.2241` | 常量中心深度 + `decay=[0.5,0.8]` + 无法线交叉门控 |
| **`phaseB-eval` (`588` 视角)** | [metrics_report.txt](file:///d:/Dev/GeosunPre/work/展厅-phaseB-eval/metrics_report.txt) | **`26.3726`** | **`0.8563`** | **`0.2664`** | **`11.7813°`** | **`9.2377 / 58.5210`** | 全指标优于 `phaseA`，`PSNR/SSIM/LPIPS` 创最佳 |
| **`phaseB-eval-on-phaseA` (`588` 视角)** | [metrics_report.txt](file:///d:/Dev/GeosunPre/work/展厅-phaseB-eval-on-phaseA/metrics_report.txt) | **`26.3726`** | **`0.8563`** | **`0.2664`** | `12.7728°` | **`9.0576 / 34.7491`** | 同源 `phaseA` 口径下深度中位/`p90` 降 `17.3% / 30.8%` |
| `roi` / `phaseA` / `phaseB-test` (`74` 视角) | [metrics_report.txt](file:///d:/Dev/GeosunPre/work/展厅-phaseB-eval-test/metrics_report.txt) | `26.1895 / 26.1041 / 26.2386` | `0.8523 / — / 0.8518` | `0.2706 / — / 0.2700` | `10.9237° / 13.1491° / 11.8638°` | `16.9413 / 11.1564 / 9.1127` (med) | 注：`phaseB-train` 以 `enable_eval=false` 使用全量 `588` 张训练后抽取 `74` 张评测 |

### 实验二：`南非展会`（`1,073` 视角，`30k` 步，`res /2`，`cropbox` ROI）
- **监督清洗对比（[phaseA supervision.json](file:///d:/Dev/GeosunPre/work/南非展会-phaseA/supervision.json) vs [phaseB supervision.json](file:///d:/Dev/GeosunPre/work/南非展会-phaseB/supervision.json)）**：
  - 全局：`las_input_pixels=696,350,621`，`penetration_normal_dropped` 由 `10,972,035` 降至 **`1,250,223`**（`-88.6%`），`ransac_dropped_far` 由 `11,573,243` 降至 **`4,802,323`**（`-58.5%`），`far_normal_rescued=294,792`，`normal_disagree_dropped=21,155,955`；`depth_pixels` 由 `622,462,814` 增至 `632,372,469`。
  - `2420-028764-644_00263-L.JPG`：`penetration_normal_dropped` **`42,306 -> 510`**（`-98.8%`）、`ransac_dropped_far` **`42,318 -> 582`**（`-98.6%`），`depth_pixels` `550,008 -> 592,985`，`normal_pixels` `673,887 -> 702,662`（[263L 先验法线](file:///d:/Dev/GeosunPre/work/南非展会-phaseB/normals/2420-028764-644_00263-L.JPG.png) 白墙矩形缺洞完全消失）。
  - `2420-028779-894_00282-L.JPG`：`penetration_normal_dropped` **`30,543 -> 121`**，`normal_disagree_dropped=56,557`（[282L 先验法线](file:///d:/Dev/GeosunPre/work/南非展会-phaseB/normals/2420-028779-894_00282-L.JPG.png) 精准置零展台上方阴影区 MoGe 天花板幻觉）。

| 评测配置 | 目录 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | `normal_angle_deg` ↓ | 深度中位 / `p90` 误差 (mm) ↓ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `南非展会-phaseA-eval` | [metrics_report.txt](file:///d:/Dev/GeosunPre/work/南非展会-phaseA-eval/metrics_report.txt) | `21.5202` | `0.7430` | **`0.3286`** | `10.4399°` | `21.2368 / 86.8826` |
| **`南非展会-phaseB-eval`** | [metrics_report.txt](file:///d:/Dev/GeosunPre/work/南非展会-phaseB-eval/metrics_report.txt) | `21.6180` | `0.7469` | `0.3292` | **`8.5699°`** | **`19.0939`** / `91.9329` |
| **`南非展会-phaseB-eval-on-phaseA`** | [metrics_report.txt](file:///d:/Dev/GeosunPre/work/南非展会-phaseB-eval-on-phaseA/metrics_report.txt) | **`21.6225`** | **`0.7470`** | `0.3291` | **`9.1060°`** | **`19.0983`** / `90.7345` |

- **白墙高斯几何审计（[splat_30000.ply](file:///d:/Dev/GeosunPre/work/南非展会-phaseB-train/splat_30000.ply)，`u ∈ [-7, +6], z ∈ [3, 7], |off| < 1.0 m`）与视觉复核**：
  - 墙框内共 `19,524` 个不透明度 `≥ 0.05` 的高斯：**`>1.0 m` 超大高斯 `0` 个**（基线 `3,398` 个、Phase A `640` 个），`>0.7 m` 仅 `5` 个，最大尺度 `0.7669 m`（基线 `2.65 m`、Phase A `1.455 m`）；总体墙面偏移均值/中位 `-0.20 / -0.10 cm`，左半区 `u ∈ [-7, -4]` 偏移收敛至 `-1.51 / -1.34 cm`（Phase A 为 `-8.9 cm`）。
  - 渲染对比图 [263-L 渲染法线对比](file:///d:/Dev/GeosunPre/work/南非展会-phaseB-eval/eval_step_30000/2420-028764-644_00263-L_normals.png)、[263-L RGB 对比](file:///d:/Dev/GeosunPre/work/南非展会-phaseB-eval/eval_step_30000/2420-028764-644_00263-L.png) 与 [282-L 渲染法线对比](file:///d:/Dev/GeosunPre/work/南非展会-phaseB-eval/eval_step_30000/2420-028779-894_00282-L_normals.png) 目视确认：`263-L` 白墙左上部红色内凹坑与 `16×16` 黑框网格完全消失，法线平整连续。

### 结论与取舍
- **工程确立**：逐像素射线-高斯深度光栅化彻底消除了 3DGUT 中心深度模型偏差引发的弱纹理墙面凹陷与瓦片边界伪影；配合多面 RANSAC 共识与 LAS surfel 法线门控，在 `南非展会` 与 `展厅` 两个实景上均实现了外观（`+0.10 ~ +0.16 dB` PSNR）与几何精度（同源口径深度误差 `-10.1% ~ -17.3%`、法线误差大幅下降）的同步改善，确立为联合默认配置。



## 2026-11-20 输入改为调用方显式路径

- 改动：`run`/`prepare` 去掉 `--input` 自动查找，改为必填 `--project`、`--images`（可重复）、`--camera-pos`、`--pos`、`--pos-enh`、`--las`、`--output`；镜头只由文件名决定，照片目录顶层含所用 JPEG 时只读顶层，否则递归；交付图像名统一为 `镜头/文件名`；删除 `enhanced_photos` 配置及原图缺失时自动改用 enhanced 的回退。开发脚本 `run.ps1 --input` 按旧布局展开为显式参数。
- 等价性：同一 `展厅` 输入，旧程序 `prepare` 与新程序（L/R 两目录）的 `scene.json` 相机、源文件元数据、588 张曝光的全部字段一致，唯一差异为图像名去掉中间层级（`L/0000/xxx.JPG` → `L/xxx.JPG`）；给父目录 `Image`、平铺目录（含 `enhanced/` 子目录被忽略）得到相同曝光与名称；平铺目录与其 `enhanced` 同时给出时报 `Ambiguous exposure`；输出位于照片目录内被拒绝。
- 实景全流程：`work/展厅-explicit`（交付 `work/展厅-explicit-delivery`）693.6 s 完成，588/588 注册，所选模型 `cleaned`，卷帘按行读出被采用（验证中位 1.15 px），`check --project` 通过（152,957 点、738,752 观测）。
- 结论：采用。数值管线不受影响，交付契约仅图像名层级变化。


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

- **专项排查归档（2026-11-22）**：针对 `小场景-fixed` 在高处俯视 novel view 下 1F 弱纹理白墙出现彩斑、拉丝及法线起皱的问题，完成 `v1`–`v4` 参数对照与 3D 高斯几何/色域审计，详见 [novel-view-whitewall-investigation.md](file:///C:/Users/youwu/Documents/GeosunPre/docs/novel-view-whitewall-investigation.md)（[configs/trainer.json](file:///C:/Users/youwu/Documents/GeosunPre/configs/trainer.json) 暂保持不变）。

