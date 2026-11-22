# `小场景-fixed` Novel View 弱纹理白墙伪影与法线起皱排查记录

- **日期**：2026-11-22
- **排查对象**：
  - 数据集：[小场景-fixed](file:///C:/Users/youwu/Documents/GeosunPre/work/1122/小场景-fixed)（`296` 帧异步双鱼眼，`296/296` 注册，地面轨迹 `Z ∈ [-0.05, 1.00] m`，启用 `rolling_shutter.json` 与 `roi.json`）
  - 基线训练与评测：[小场景-fixed-train](file:///C:/Users/youwu/Documents/GeosunPre/work/1122/小场景-fixed-train/project.licht)、[小场景-fixed-eval](file:///C:/Users/youwu/Documents/GeosunPre/work/1122/小场景-fixed-eval/metrics_report.txt)
  - 对应训练配置（未改动）：[configs/trainer.json](file:///C:/Users/youwu/Documents/GeosunPre/configs/trainer.json)
- **核心结论摘要**：本轮定位并确认了多项引起白墙超色域彩斑、细长拉丝与法线起皱的参数与物理诱因，`v2`–`v4` 虽已基本消除超色域色斑并把全局表面法线误差由 `10.61°` 降至 `9.23°–9.69°`、1F 细长针由 `52.9%` 降至 `1.2%–6.5%`，但高处俯视 novel view 下 1F 弱纹理白墙仍有灰白斑驳（尤其最左第 1 扇黑窗左侧及 1F 窗间），主要受限于跨视角死白过曝截断、近距未遮蔽移动阴影及 `erank_reg / scale_reg` 压针与空隙微孔的权衡，现完整归档以便后续继续深入。

---

## 1. 问题现象

在 LichtFeld Studio（及后续 Spirula）切换到高处俯视或斜掠的非训练视角（novel view）后：
1. **RGB 视图**：贴近镜头或斜掠角度的一楼弱纹理白墙（门洞白墙、三窗白墙及最左侧窗左白墙）出现雾状灰白模糊、粉红/青绿彩色斑驳，以及斜向拉丝与亮白发光划痕；而同视角下的**灰色地砖**与**二楼上层白墙**明显更干净锐利。
2. **法线视图**：基线模型下白墙表面法线呈现类似“揉皱锡纸”的不规则凹凸波纹；同时左侧黑色玻璃电梯柱前缘在贴近观察时存在白色絮状悬浮物。

---

## 2. 已排除假设与预处理核验

在检查训练器产物前，先对上游预处理 [小场景-fixed](file:///C:/Users/youwu/Documents/GeosunPre/work/1122/小场景-fixed) 及渲染/几何边界逐项核验，**排除了预处理输入质量问题**：

1. **LAS 初始化点云平面精度正常**（[points3D.ply](file:///C:/Users/youwu/Documents/GeosunPre/work/1122/小场景-fixed/sparse/0/points3D.ply)）：
   - 左侧白墙（`00079-L` 前，距相机 `0.81 m`）：RANSAC 平面内点 `2,114 / 2,545`，距离中位数 `1.50 mm`（`p90 = 3.28 mm`；全区拟合 `med = 1.36 mm`、`p90 = 3.05 mm`）；
   - 门洞白墙（`00088-L` 前，距相机 `1.99 m`）：`med = 1.10 mm`（`p90 = 5.53 mm`）；
   - 三窗白墙（`00051-L` 前，距相机 `0.93 m`）：`med = 1.75 mm`，沿墙分段 `u ∈ [-10, +6] m` 平面偏差中位数仅 `-2.6 ~ +2.2 mm`（说明各帧定位与 LAS 无墙体错位或重影分层）；
   - 1F 纯白墙补丁内 `off < -3 cm` 的点（LAS 占比 `415 / 1,514 = 27.4%`，集中在 `-3 ~ -13.4 cm`）经核对为墙背侧门窗内沿及室内走廊真实墙面（如 `00151-L ~ 00167-L` 进入室内段所见）。
2. **单目/LAS 深度与法线先验本身干净**（`depths/`、`normals/`）：
   - `00079-L`、`00088-L` 纯白墙区域深度有效率 `98.6%`、`99.9%`，局部二阶差分 `|dxx|` 中位数 `0 ~ 1 mm`（`p90 = 1 mm`）；
   - 法线先验与 LAS 真实墙面法线夹角中位数仅 `1.08° ~ 1.51°`（`p90 = 2.02° ~ 2.50°`），不存在褶皱。
3. **训练视角表面得分正常**：基线 `v1` 在近距看白墙的训练帧 `00079-L ~ 00088-L` 上 PSNR 达到 `27.31 ~ 29.44 dB`。
4. **非光栅化后端单一原因**：用户将 LichtFeld Studio 光栅化后端由 `3DGS` 切换至 `3DGUT` 并调整焦距至 `20.8 mm` 后，1F 三窗白墙的粉红/青绿糊斑与亮白竖向划痕依然显著。
5. **三窗白墙前方并非空气浮云**：
   - 在 `v1` 三窗白墙正前方 `off ∈ [0.05, 1.2] m, Z ∈ [0.8, 2.3] m, u ∈ [-6.5, 6.5] m` 空气盒内 `opac >= 0.02` 高斯数为 `0`；
   - 对 `v2` 模型发射 novel view 3DGUT 穿透射线（至 `u = -7.5 / -6.0 / -3.5 m`），射线在墙面 `±1 cm` 内直接穿透 **`57 ~ 88` 个重叠高斯**，前方空中并无遮挡浮云；
   - 例外：画面最左侧黑色玻璃电梯柱旁（`off ≈ 3.33 m, z ≈ 2.34 m`）确有约 `12,024` 个亮白悬浮高斯（`luma > 0.65`），属于深色玻璃幕墙的镜面反射虚像，与白墙主体斑驳属两个独立问题。
6. **非 `SH1` 方向外推主导**：对 `v2` 离线将 `f_rest_0..8` 置零生成的 [v2_no_big_needle_sh0.ply](file:///C:/Users/youwu/Documents/GeosunPre/work/1122/diag-filter/v2_no_big_needle_sh0.ply) 与保留 `SH1` 的 [v2_no_big_needle.ply](file:///C:/Users/youwu/Documents/GeosunPre/work/1122/diag-filter/v2_no_big_needle.ply) 视觉表现几乎无差异。

---

## 3. 核心根因定位

### 3.1 `dc_reg = 0.0` + 局部双边网格掩盖导致的超 `[0, 1]` 色域发散与互补抵消（已由 `v2` 验证）
- 基线 `v1` 设置 `dc_reg = 0.0`、`use_exposure_correction = true`（每图独立 `16×16×8` 双边网格）、`opacity_decay = 0.002`。
- 审计显示：`v1` 全模型 **`38.97%`** 的高斯超出 `[0, 1]` 合法色域，三窗白墙表面（`±5 cm`、`N = 193,639`）高达 **`43.10%`（`83,459` 个）**超色域，其中 `RGB > 1.2` 有 `36,398` 个、`RGB < -0.2` 有 `6,724` 个，RGB 极值达 **`[-1.31, +2.75]`**；而 LAS 点云在同区域高饱和点（`sat > 0.15`）仅 `0.14%`。
- 在低处训练视角与每图双边网格保护下，正/负超色域高斯与红/青互补色高斯按固定视角排序混合后彼此抵消；切到高处 novel view 后深度排序翻转，直接爆出粉红/青绿斑块与亮白划痕。

### 3.2 `normal_prior_decay` 过早衰减与 `normal_consistency_weight` 倒挂导致法线起皱
- 基线 `v1` 中 `normal_prior_decay_start = 0.6`（`18k` 步起外部先验余弦衰减，`30k` 归零），而自一致项 `normal_consistency_weight = 0.30` 全程不衰减（是法线先验 `0.05` 的 `6×`、法线深度先验 `0.02` 的 `15×`）。
- 失去外部先验锚定后，纯白墙极微小的深度起伏即可通过 `normal_consistency` 与自一致梯度锁定为大面积起伏褶皱。

### 3.3 近距 1F 白墙与 2F 干净白墙的同立面 A/B 审计与离线过滤实锤
在 `v2` 模型上取三窗墙面同宽 `5.2 m²`（`u ∈ [-9.5, -5.5] m`）的**发糊 1F 左红框墙**与**正上方干净 2F 墙**直接对比：

| 审计指标（`v2` 模型，`5.2 m²`） | **1F 左红框白墙（发糊）** | **2F 正上方白墙（干净）** | 倍比 |
| :--- | :---: | :---: | :---: |
| 高斯数量 / 密度 | `26,407`（`5,078 / m²`） | `4,261`（`819 / m²`） | **`6.20×`** |
| **极细长针占比（`max/mid > 5` / `> 10`）** | **`50.4%` / `24.1%`** | `18.0%` / `2.7%` | **`9×`**（`>10`） |
| **`> 30 cm` 超大高斯数** | **`432`**（`p99 = 41.0 cm`、`max = 74.0 cm`） | `45`（`p99 = 31.6 cm`） | **`9.6×`** |
| 过绘层数（不透明度加权面积 / 墙面积） | **`6.20×`** | `3.93×` | `1.58×` |
| 墙面暗色杂点（`luma p5` / `std`） | `p5 = 0.12` / `std = 0.278` | `p5 = 0.55` / `std = 0.157` | — |
| 单高斯最短轴法线 vs 墙面夹角（`med / p75`） | **`31.3° / 58.1°`** | `17.5° / 40.7°` | `1.79×` |

- **离线过滤对照**（见 `work/1122/diag-filter/`，剔除 `max_scale > 0.15 m` 或 `max/mid > 5.0`，全局保留 `2,179,284 / 2,999,777 = 72.65%`）：
  - **灰色地砖区域 100% 完好无孔**；
  - **1F 左红框白墙覆盖率由 `6.20×` 骤降至 `0.86×`（`12,823 / 26,407`）**，整面墙暴露出密集黑洞与 `9.5%`（`1,214` 个）`luma < 0.20` 的纯黑小高斯。这证明拉丝糊斑确实主要由细长针与大薄片在 `6.2×` 多层堆叠下视角排序翻转所致，但离线删点会露底，必须训练期约束。

### 3.4 目前尚未彻底消除的物理瓶颈（也是 `v3`/`v4` 效果仍不尽如人意的原因）
1. **跨视角死白过曝截断与未遮蔽贴墙阴影**：
   - 将第 1 扇窗左侧残余糊斑中心 `P_smudge = (2.539, 0.328, 1.800)` 反投影回训练帧：`00157-R ~ 00163-R`（暗室内走廊透过门洞向外拍摄）把该白墙完全过曝截断为 `(255, 255, 255)`，而室外 `00170-L ~ 00171-L` 在 `1.24–1.26 m` 近距拍得 `(165, 169, 170)` 与 `(127, 127, 125)`（`00168-L` 为 `(224, 238, 241)`）。
   - 已发生饱和截断的 `255` 像素与近距采集者投射到墙面的柔和暗影均无法被全局曝光模型 `PPISP` 还原，迫使优化器继续在局部堆叠 `5.4×–6.1×` 明暗互补的圆盘与暗点（`luma < 0.20` 占 `7.7%–8.2%`），在 novel view 下呈现灰白斑驳。
2. **现有正则化对“压细长针”与“缩尺穿孔/球化”的耦合冲突**：
   - `erank_reg`（[effective_rank.cuh](file:///c:/Users/youwu/Documents/LichtFeld-Studio/src/training/include/lfs/training/effective_rank.cuh#L16-L56)）对 `max/mid > 1` 施压提高有效秩至 `2.0` 时，在 `gut` 路径下会同时把最薄轴 `s_min` 从 `0.031 mm` 推厚至 `0.58–1.16 mm`（降低扁平度、钝化单高斯法线）；
   - 同时在 `gut = true` 下 `scale_reg`（[trainer.cpp](file:///c:/Users/youwu/Documents/LichtFeld-Studio/src/training/trainer.cpp#L2272-L2278) / [regularization.cu](file:///c:/Users/youwu/Documents/LichtFeld-Studio/src/training/kernels/regularization.cu)）未应用衰减且直接对三轴 `exp(scaling_raw)` 求和惩罚，提高 `erank_reg` 和 `scale_reg` 易把圆盘压得过小（`1.24–1.46 cm`），在 Spirula `alpha` 视图和 `eval` 中表现为墙脚/砖缝浅灰半透明微孔（`alpha < 0.98` 由 `28.4%` 升至 `34.3%–36.9%`，整体 `Bias` 偏暗 `-0.03`）。

---

## 4. `v1 ~ v4` 实验配置与实测对照汇总

所有实验均在同台机器（`RTX 5070 Ti`）、同数据集 [小场景-fixed](file:///C:/Users/youwu/Documents/GeosunPre/work/1122/小场景-fixed) 上跑完 `30,000` 步与 `eval --eval-all`（296 视角）；`v2`、`v3`、`v4` 输出目录均保留在 `work/1122/` 下，原始 [configs/trainer.json](file:///C:/Users/youwu/Documents/GeosunPre/configs/trainer.json) 未作任何修改。

### 4.1 相对基线 `configs/trainer.json` 的参数差异

| 参数 | `v1`（基线） | `v2`（色域+法线） | `v3`（强压针+缩尺） | `v4`（回调平衡） |
| :--- | :---: | :---: | :---: | :---: |
| `dc_reg` | `0.0` | `0.005` | `0.005` | `0.005` |
| `use_exposure_correction` / `use_ppisp` | `true` / `false` | `false` / `true` | `false` / `true` | `false` / `true` |
| `sh_degree` / `sh_rest_reg` | `0` / `0.0` | `1` / `0.001` | `0` / `0.0` | `0` / `0.0` |
| `normal_prior_decay_start` | `0.6` | `0.95` | `0.95` | `0.95` |
| `normal_loss_weight` / `normal_prior_depth_weight` | `0.05` / `0.02` | `0.1` / `0.05` | `0.1` / `0.05` | `0.1` / `0.05` |
| `normal_consistency_weight` | `0.3` | `0.1` | `0.1` | `0.1` |
| `means_noise_weight` / `stop_refine` | `50.0` / `28500` | `15.0` / `25000` | `15.0` / `25000` | `15.0` / `25000` |
| `opacity_decay` | `0.002` | `0.0005` | `0.0005` | `0.0002` |
| `depth_distortion_weight` | `0.25` | `0.5` | `1.0` | `0.5` |
| `erank_reg` | `0.0` | `0.0` | `0.02` | `0.005` |
| `scale_reg` / `scale_reg_decay_power` | `0.01` / `0.4` | `0.01` / `0.4` | `0.02` / `-1.0` | `0.005` / `-1.0` |
| `max_screen_share` | `0.5865` | `0.5865` | `0.15` | `0.25` |
| `gradient_loss_weight` | `1.8` | `1.8` | `0.2` | `0.2` |

### 4.2 全视角 `eval --eval-all`（296 视角）量化对比

| 指标 | `v1`（[报告](file:///C:/Users/youwu/Documents/GeosunPre/work/1122/小场景-fixed-eval/metrics_report.txt)） | `v2`（[报告](file:///C:/Users/youwu/Documents/GeosunPre/work/1122/小场景-fixed-eval-v2/metrics_report.txt)） | `v3`（[报告](file:///C:/Users/youwu/Documents/GeosunPre/work/1122/小场景-fixed-eval-v3/metrics_report.txt)） | `v4`（[报告](file:///C:/Users/youwu/Documents/GeosunPre/work/1122/小场景-fixed-eval-v4/metrics_report.txt)） |
| :--- | :---: | :---: | :---: | :---: |
| 训练耗时 / 最终高斯数 | — / `2,978,391` | `909.88 s` / `2,999,777` | `814.03 s` / `2,990,974` | `822.90 s` / `2,975,349` |
| PSNR ↑ | `24.3954` | **`24.9931`** | `24.1884` | `24.4614` |
| Aligned PSNR ↑ | `25.4975` | **`26.4003`** | `25.4583` | `25.7846` |
| SSIM / LPIPS | `0.8314` / `0.2407` | **`0.8384` / `0.2325`** | `0.8230` / `0.2470` | `0.8290` / `0.2437` |
| Raw Bias | `(0.041, 0.041, 0.046)` | **`(-0.006, -0.009, 0.001)`** | `(-0.032, -0.030, -0.027)` | `(-0.030, -0.028, -0.026)` |
| 透明缺口（`alpha < 0.98`）↓ | `0.3194` | **`0.2844`** | `0.3685` | `0.3431` |
| 渲染法线角误差 ↓ | `10.6135°` | **`9.2327°`** | `10.0364°` | `9.6880°` |
| 深度误差（`AbsRel` / `med` / `p90`）↓ | `0.0053` / `6.24` / `34.89 mm` | `0.0049` / **`5.59`** / `31.92 mm` | `0.0050` / `5.93` / **`31.02 mm`** | `0.0052` / `6.00` / `32.92 mm` |

### 4.3 3D 高斯白墙几何、细长针与色域审计对比

| 审计项 | `v1` | `v2` | `v3` | `v4` |
| :--- | :---: | :---: | :---: | :---: |
| 全局超 `[0, 1]` 色域占比 | `38.97%` | **`0.14%`** | `0.46%` | `0.46%` |
| 三窗白墙（`±5 cm`）超色域占比 | `43.10%`（`83,459`） | **`0.23%`**（`497`） | `0.46%`（`1,030`） | `0.42%`（`925`） |
| 全局细长针（`max/mid > 5` / `> 10`） | `25.0%` / `8.0%` | `26.1%` / `8.7%` | **`1.1%` / `0.3%`** | `4.7%` / `1.1%` |
| 1F 左红框细长针（`max/mid > 5` / `> 10`） | `52.9%` / `25.5%` | `50.4%` / `24.1%` | **`1.2%` / `0.4%`** | `6.5%` / `1.5%` |
| 1F 左红框 `>15 cm` / `>30 cm` 薄片 | `1,134` / `588` | `831` / `432` | **`444` / `262`** | `552` / `341` |
| 1F 左红框过绘 / 暗点（`luma < 0.20`） | `6.10×` / `10.9%` | `6.20×` / `7.9%` | `5.39×` / `7.7%` | `6.07×` / `8.2%` |
| 其余白墙 `>25 cm`（左 / 门 / 右 / 2F） | `205 / 342 / 155 / 323` | `55 / 194 / 92 / 153` | **`10 / 26 / 36 / 23`** | `44 / 111 / 74 / 114` |
| 全局尺度中位 `s_min` / `s_mid` / `s_max` | `0.036 mm` / `0.70` / `2.28 cm` | `0.031 mm` / `0.60` / `2.01 cm` | `1.157 mm` / `0.80` / `1.24 cm` | `0.576 mm` / `0.83` / `1.46 cm` |
| 扁平比（`s_min/s_mid < 0.1`） / 球化（`> 0.5`） | `66.8%` / — | `68.0%` / `12.0%` | `43.1%` / `18.4%` | `52.2%` / `16.7%` |
| 墙面不透明度中位（左 / 门 / 右 / 2F） | `0.53 / 0.55 / 0.53 / 0.70` | `0.71 / 0.73 / 0.81 / 0.97` | `0.70 / 0.85 / 0.91 / 0.98` | `0.66 / 0.76 / 0.86 / 0.98` |

### 4.4 本地保留产物路径索引
- `v2`：[小场景-fixed-train-v2](file:///C:/Users/youwu/Documents/GeosunPre/work/1122/小场景-fixed-train-v2/project.licht) / [小场景-fixed-eval-v2](file:///C:/Users/youwu/Documents/GeosunPre/work/1122/小场景-fixed-eval-v2)
- `v3`：[小场景-fixed-train-v3](file:///C:/Users/youwu/Documents/GeosunPre/work/1122/小场景-fixed-train-v3/project.licht) / [小场景-fixed-eval-v3](file:///C:/Users/youwu/Documents/GeosunPre/work/1122/小场景-fixed-eval-v3)（Spirula 目视确认 `depth_normal` 墙面已完全平整、粉红/青绿糊斑及亮白划痕消除，但 `alpha` 视图下墙脚与砖缝出现细密半透明微孔、左 1 窗左侧仍留灰白斑驳）
- `v4`：[小场景-fixed-train-v4](file:///C:/Users/youwu/Documents/GeosunPre/work/1122/小场景-fixed-train-v4/project.licht)（含 [splat_30000.ply](file:///C:/Users/youwu/Documents/GeosunPre/work/1122/小场景-fixed-train-v4/splat_30000.ply)） / [小场景-fixed-eval-v4](file:///C:/Users/youwu/Documents/GeosunPre/work/1122/小场景-fixed-eval-v4)
- 离线过滤对照：[v2_no_big_needle.ply](file:///C:/Users/youwu/Documents/GeosunPre/work/1122/diag-filter/v2_no_big_needle.ply)、[v2_no_big_needle_sh0.ply](file:///C:/Users/youwu/Documents/GeosunPre/work/1122/diag-filter/v2_no_big_needle_sh0.ply)

---

## 5. 后续深入调查方向

1. **屏蔽大光比死白过曝截断像素**：在室内外穿行采集场景中，室内往外看把墙面截断为 `RGB ≈ 255` 的像素（如 `00157-R ~ 00163-R`）与近距室外帧存在不可调和的物理冲突；应评估在训练掩码或光度损失中过滤高光截断区域（例如大面积 `min(R,G,B) >= 252` 仅保留室外非饱和监督）。
2. **抑制近距弱纹理表面因软阴影/微闪烁引发的过度裂变与过绘**：调查 MRNF 致密化在极近距（`< 1.5 m`）无纹理墙面上的拆分阈值与局部模糊/软阴影权重，避免同一物理平面长出 `5,000+ splats/m²` 与 `6×` 多层互补叠片；同时排查 `v4` 在恢复 `gradient_loss_weight` / `sh_degree` 对照中对近距白墙高频过拟合与远区锐度的各自贡献。
3. **改进平片各向异性正则（保扁平、只限长宽比）**：`erank_reg` 当前同时惩罚面内长短轴比并间接推厚 `s_min`；若改为仅惩罚面内两主轴比（`max_scale / mid_scale`）而不抬高 `s_min`，同时修复 `gut` 路径下 `scale_reg` 未按 `scale_reg_decay_power` 衰减或按尺度上限软阈值的行为，有望在不产生 `alpha` 微孔和偏暗的前提下消除细长针。
4. **弱纹理平面强几何绑定 / LAS 平面正则**：利用已有毫米级精度（`1.1–1.7 mm`）的 `points3D.ply` 与 `normals/`，针对非边缘弱纹理平面约束高斯只做切平面内铺满并降低面外漂移与叠层，同时独立处理深色玻璃幕墙的镜面反射虚像。

---

## 6. 训练动态演化新发现（3,000 → 16,000 步逐步劣化）与源码机制定位

### 6.1 实时训练过程观测事实（`小场景-las-norm-out` + 基线 `configs/trainer.json`）

在最新预处理数据集 [小场景-las-norm-out](file:///C:/Users/youwu/Documents/GeosunPre/work/小场景-las-norm-out)（启用与 `小场景-fixed` 相同的 `max_cap = 3,000,000`、`sh_degree = 0`、`use_exposure_correction = true`、`dc_reg = 0.0`、`gradient_loss_weight = 1.8`、`opacity_decay = 0.002` 等默认配置）上实时观察 LichtFeld Studio 视口高处俯视 novel view 随迭代步数的变化：
- **`3,250` 步**：活动高斯数已打满 `3,000,000` 上限，但 1F 窗左与窗间白墙均匀干净，几乎无斑驳、斜向拉丝或粉红/青绿彩斑；
- **`10,100` 步**：1F 窗左与窗间白墙开始显现灰白斑驳与竖向拉丝；
- **`16,080` 步**：1F 窗左出现明显粉红/青绿彩斑、亮白竖向划痕与雾状灰白糊斑；同视角下 2F 上层白墙全程保持干净。

**核心推论**：上述时间线进一步独立排除了 LAS 初始化、相机位姿、法线先验以及前 `3,000` 步生长阶段把白墙“长歪”的可能性——白墙在刚满 `3M` 时本身是干净平滑的，问题直接源自 **`3,000` 步之后才逐步积累放大的持续优化动力学**。

### 6.2 源码核验定位到的四项 `3k → 16k` 劣化机制

对照 LichtFeld Studio 训练源码，基线配置下以下四股劣化力量均在 `1,000 ~ 2,400` 步刚启动，因此在 `3,250` 步尚未明显破坏白墙，却在 `3,250 → 10,100 → 16,080` 步持续累积：

1. **`ExposureChroma` 逐图双边网格过预热 + `dc_reg = 0.0` 诱发色域发散**
   - `exposure_correction_grid_start_iter = 1000` 启用逐图 `16×16×8` `ExposureChroma` 双边网格（[bilateral_grid_forward.cu](file:///C:/Users/youwu/Documents/LichtFeld-Studio/src/training/kernels/bilateral_grid_forward.cu#L222-L240)），且自身带 `BilateralGridConfig::warmup_steps = 1000` 线性预热（[bilateral_grid.hpp](file:///C:/Users/youwu/Documents/LichtFeld-Studio/src/training/components/bilateral_grid.hpp#L39-L41)）。
   - 296 张训练图到 `3,250` 步时平均每张仅更新 `(3250 - 1000) / 296 ≈ 7.6` 次（前半仍在预热），网格接近恒等；到 `10,100` / `16,080` 步平均每张更新约 `31` / `51` 次后，网格充分吸收每图局部色度偏差，叠加 GUT SH 前向评估不截断 `[0, 1]`（[SphericalHarmonicsCUDA.cu](file:///C:/Users/youwu/Documents/LichtFeld-Studio/src/training/rasterization/gsplat/SphericalHarmonicsCUDA.cu#L420)）、截断仅发生在外观校正之后（[trainer.cpp](file:///C:/Users/youwu/Documents/LichtFeld-Studio/src/training/trainer.cpp#L7949)）且 `dc_reg = 0.0`，跨视角过曝/阴影冲突在 `3k → 16k` 期间迅速放大为单高斯超色域与红/青互补色斑。

2. **MRNF 满额后的“无误差盲拆分 + GUT 全局不透明度衰减”叠片循环**
   - [mrnf.cpp](file:///C:/Users/youwu/Documents/LichtFeld-Studio/src/training/strategies/mrnf.cpp#L531-L532) 明确注释并记录：`opacity_decay_rendered_only` 在 `gut = true` 下**不生效**。每 `refine_every = 163` 步，[mrnf_kernels.cu](file:///C:/Users/youwu/Documents/LichtFeld-Studio/src/training/kernels/mrnf_kernels.cu#L347-L361) 的 `mrnf_decay_kernel` 对全部 `3M` 高斯统一扣减 `opacity_decay * (1 - t)`。
   - 一旦像本次这样在 `3,250` 步就顶满 `max_cap = 3,000,000` 且 `growth_exchange = false`，[mrnf.cpp](file:///C:/Users/youwu/Documents/LichtFeld-Studio/src/training/strategies/mrnf.cpp#L1504-L1578) 中 `budget == pruned_count == requested_replace`，基于误差图的生长 `n_grow = 0`；全部被衰减剪出的空位转由 `replace_parent_weights_kernel`（[mrnf_kernels.cu](file:///C:/Users/youwu/Documents/LichtFeld-Studio/src/training/kernels/mrnf_kernels.cu#L63-L83)）分配。
   - `replace_parent_weights_kernel` 权重仅为 `opacity * (visibility > 0) * edge_factor`，**完全不看像素误差图**；高频出镜、不透明度高的近距 1F 白墙在 `3,250 → 16,080` 步的约 `79` 轮 `refine` 中反复中签做长轴一分为二（每次将子代不透明度降为 `0.6×`，[densification_kernels.cu](file:///C:/Users/youwu/Documents/LichtFeld-Studio/src/training/kernels/densification_kernels.cu#L155-L157)），把原本干净的单层墙面持续拆叠成多层半透明薄片。

3. **`gradient_loss_weight = 1.8` 于 `2,000` 步启动并绕过外观校正直接作用在 `raw_render`**
   - [gradient_residual.hpp](file:///C:/Users/youwu/Documents/LichtFeld-Studio/src/training/kernels/gradient_residual.hpp#L11) 定义 `GRADIENT_LOSS_START_STEP = 2000`；当启用外观模型时，[trainer.cpp](file:///C:/Users/youwu/Documents/LichtFeld-Studio/src/training/trainer.cpp#L1903-L1907) 与 [trainer.cpp](file:///C:/Users/youwu/Documents/LichtFeld-Studio/src/training/trainer.cpp#L8869-L8875) 直接拿未经 PPISP/双边网格校正且未截断的 `raw_render` 与原图 `gt` 计算亮度 Sobel 梯度残差，在 `2k → 16k` 持续将近距移动阴影梯度、过曝 `255` 截断边及曝光倍率差强刻进白墙原始高斯。

4. **扁平化与法线项于 `2,400` 步启动 + `3k → 15k` 长宽比/屏幕占比约束空窗 + 深度权重衰减**
   - `normal_start_fraction = 0.08`（`2,400` 步）开启 `normal_flatten_weight = 0.1`（[GeometryFeatures.cu](file:///C:/Users/youwu/Documents/LichtFeld-Studio/src/training/rasterization/gsplat/GeometryFeatures.cu#L72-L78)）与 `normal_consistency_weight = 0.3`；同时基线 `erank_reg = 0.0`，而 `screen_share_shrink_active`（[mrnf.cpp](file:///C:/Users/youwu/Documents/LichtFeld-Studio/src/training/strategies/mrnf.cpp#L2717-L2724)）要到 `grow_until_iter = 15,000` 步之后才激活，metric 深度权重则按 `2.0 * 0.25^(iter/30000)`（[trainer.cpp](file:///C:/Users/youwu/Documents/LichtFeld-Studio/src/training/trainer.cpp#L8218-L8220)）由 `1.74` 持续衰减至 `0.95`。
   - 另：`normal_prior_decay_start = 0.6`（`18,000` 步）在 `16,080` 步尚未触发，确认 `3k → 16k` RGB 斑驳拉丝与 `18k` 之后法线起皱属两条先后叠加的链路。

### 6.3 隔离对照实验设计与实测结果（`work/las-norm-ablation/`）

为在 [小场景-las-norm-out](file:///C:/Users/youwu/Documents/GeosunPre/work/小场景-las-norm-out) 上将“外观/原始梯度”与“满额后 MRNF 无误差盲拆分+全局衰减”两股贡献一刀切开，设计 3 组对照实验（均跑满 `30,000` 步，并在 `3000 / 10000 / 16000` 步保存独立 `.licht` 快照转 PLY，辅以统一高处俯视 novel view 渲染与 30k 全视角 `eval --eval-all`）：
- `exp0-baseline`：基线 [configs/trainer.json](file:///C:/Users/youwu/Documents/GeosunPre/configs/trainer.json) 完整复现；
- `exp1-clean-app`：仅关闭局部双边网格并切换 PPISP + 约束色域与原始梯度/法线先验衰减（`use_exposure_correction=false, use_ppisp=true, dc_reg=0.005, gradient_loss_weight=0.0, normal_prior_decay_start=0.95, normal_loss_weight=0.1, normal_prior_depth_weight=0.05, normal_consistency_weight=0.1`），**保留** `opacity_decay=0.002, scale_decay=0.002, stop_refine=28500, means_noise_weight=50.0, erank_reg=0.0`；
- `exp2-freeze-topo`：在 `exp1-clean-app` 基础上**仅**切断 4k 后的盲拆分与全局衰减（`opacity_decay=0.0, scale_decay=0.0, stop_refine=4000`）。

实验配置与产物统一保留在 [work/las-norm-ablation/](file:///C:/Users/youwu/Documents/GeosunPre/work/las-norm-ablation) 下：
- 配置：[exp0-baseline.json](file:///C:/Users/youwu/Documents/GeosunPre/work/las-norm-ablation/configs/exp0-baseline.json)、[exp1-clean-app.json](file:///C:/Users/youwu/Documents/GeosunPre/work/las-norm-ablation/configs/exp1-clean-app.json)、[exp2-freeze-topo.json](file:///C:/Users/youwu/Documents/GeosunPre/work/las-norm-ablation/configs/exp2-freeze-topo.json)；
- 评估报告：[exp0 metrics](file:///C:/Users/youwu/Documents/GeosunPre/work/las-norm-ablation/exp0-baseline/eval/metrics_report.txt)、[exp1 metrics](file:///C:/Users/youwu/Documents/GeosunPre/work/las-norm-ablation/exp1-clean-app/eval/metrics_report.txt)、[exp2 metrics](file:///C:/Users/youwu/Documents/GeosunPre/work/las-norm-ablation/exp2-freeze-topo/eval/metrics_report.txt)。

#### 6.3.1 30k 全视角 `eval --eval-all` 指标对比

| 实验组 | 耗时 | `#GS` (crop 后) | PSNR | SSIM | LPIPS | Aligned PSNR | Aligned SSIM | Aligned LPIPS | Bias (raw) | 法线误差 (deg) | 深度中位 / P90 (mm) | `alpha < 0.98` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| `exp0-baseline` | `843.8s` | `2,984,778` | `24.2966` | `0.8293` | `0.2415` | `25.4524` | `0.8537` | `0.2374` | `(+0.042, +0.041, +0.047)` | `9.0262` | `6.33 / 34.33` | `0.3147` |
| `exp1-clean-app` | `818.4s` | `2,981,088` | `24.4578` | `0.8285` | `0.2435` | `25.8617` | `0.8557` | `0.2385` | `(-0.033, -0.032, -0.028)` | `8.1202` | `6.47 / 33.88` | `0.3671` |
| `exp2-freeze-topo` | `804.4s` | `2,994,278` | **`24.5354`** | **`0.8294`** | **`0.2420`** | **`25.9264`** | **`0.8562`** | **`0.2370`** | `(-0.027, -0.026, -0.023)` | **`8.1186`** | `6.38 / 33.57` | `0.3198` |

注：`exp1` 与 `exp2` 将 `dc_reg` 从 `0.0` 调到 `0.005` 且切用 PPISP 后，不仅消除了色域溢出，还把 `Aligned PSNR` 提升 `+0.41 ~ +0.47 dB`、法线误差从 `9.03°` 降至 `8.12°`；`exp2` 停止 4k 后全球衰减则进一步保住了全视角覆盖率（`alpha < 0.98` 由 `0.3671` 回落至 `0.3198`）并取得三组最高 `PSNR / Aligned PSNR / SSIM`。

#### 6.3.2 3k → 10k → 16k → 30k 3D 高斯墙面审计时间线

利用拟合墙面坐标系对每组 `snap_{3000,10000,16000}.ply` 与 `splat_30000.ply` 共 12 个模型做三维体检（墙面带均取 `|off| < 5 cm`）：

| 区域 / 全局 | 实验组 | 指标演化（`3,000 → 10,000 → 16,000 → 30,000`） |
|---|---|---|
| 全局（`~2.99M`） | `exp0` | `OOG: 18.88% → 27.72% → 33.52% → 39.04%` \| `>5: 2.1% → 12.6% → 18.5% → 25.0%` \| `s_min: 5.41 → 1.45 → 0.51 → 0.036 mm` \| `flat<0.1: 6.5% → 39.8% → 54.4% → 66.9%` |
| 全局（`~2.99M`） | `exp1` | `OOG: 0.57% → 0.30% → 0.27% → 0.47%` \| `>5: 2.2% → 13.2% → 19.1% → 25.8%` \| `s_min: 4.97 → 1.24 → 0.36 → 0.017 mm` \| `flat<0.1: 7.2% → 43.5% → 58.8% → 71.3%` |
| 全局（`~2.99M`） | `exp2` | `OOG: 0.57% → 0.30% → 0.26% → 0.44%` \| `>5: 2.2% → 12.8% → 18.4% → 24.8%` \| `s_min: 4.97 → 1.36 → 0.45 → 0.024 mm` \| `flat<0.1: 6.9% → 39.1% → 55.8% → 69.2%` |
| `1F_win1_left` (`6.0 m²`) | `exp0` | `n: 37,707 → 33,012 → 30,189 → 29,077` \| `od: 3.26x → 4.31x → 4.95x → 5.38x` \| `OOG: 0.03% → 10.06% → 26.57% → 42.12%` \| `>5: 0.1% → 17.0% → 35.4% → 50.7%` \| `>30cm: 11 → 188 → 382 → 564` \| `dark: 0.0% → 0.7% → 5.4% → 10.8%` |
| `1F_win1_left` (`6.0 m²`) | `exp1` | `n: 35,991 → 32,052 → 29,312 → 28,376` \| `od: 2.71x → 3.90x → 4.58x → 5.10x` \| `OOG: 0.00% → 0.15% → 0.24% → 0.31%` \| `>5: 0.3% → 18.1% → 36.2% → 50.3%` \| `>30cm: 16 → 143 → 372 → 494` \| `dark: 0.0% → 1.1% → 4.2% → 7.7%` |
| `1F_win1_left` (`6.0 m²`) | `exp2` | `n: 36,363 → 36,438 → 36,471 → 36,477` \| `od: 2.89x → 3.86x → 4.51x → 4.84x` \| `OOG: 0.00% → 0.08% → 0.15% → 0.14%` \| `>5: 0.2% → 18.4% → 34.9% → 47.6%` \| `>30cm: 18 → 124 → 243 → 313` \| `dark: 0.0% → 1.0% → 4.6% → 8.1%` |
| `1F_between_wins` (`5.7 m²`) | `exp0` | `OOG: 2.61% → 20.85% → 32.77% → 42.32%` \| `>5: 2.0% → 27.4% → 39.0% → 47.0%` \| `>30cm: 0 → 134 → 262 → 402` \| `dark: 1.9% → 2.6% → 6.1% → 11.9%` |
| `1F_between_wins` (`5.7 m²`) | `exp1` | `OOG: 0.00% → 0.02% → 0.06% → 0.60%` \| `>5: 2.4% → 28.1% → 40.0% → 48.2%` \| `>30cm: 0 → 124 → 262 → 407` \| `dark: 2.0% → 3.8% → 8.1% → 13.2%` |
| `1F_between_wins` (`5.7 m²`) | `exp2` | `OOG: 0.01% → 0.02% → 0.04% → 0.48%` \| `>5: 2.5% → 27.8% → 39.4% → 47.6%` \| `>30cm: 0 → 123 → 229 → 349` \| `dark: 2.0% → 3.6% → 7.4% → 12.1%` |
| `2F_above_clean` (`6.0 m²`) | `exp0/1/2` | `n: ~4,600–5,551`（仅 1F 密度的 `1/6 ~ 1/8`）\| `30k` 时 `>5` 仅 `14.0%–15.0%` \| `dark` 仅 `0.28%–0.70%`（但 `exp0` 仍因未限色域达 `44.32%` OOG） |

#### 6.3.3 统一中庭高处俯视 Novel View 视觉核验

使用统一针孔探测相机（`1280×720`, `fx=fy=680`, `novel_high_left` 与 `novel_wall_closeup`）渲染全部 12 个 PLY 后直接肉眼比对（与 1:1 放大矩阵对齐）：
1. **3k 起点一致干净**：`3,000` 步时三组 1F 墙面均平滑均匀、无明显竖条拉丝与彩斑。
2. **仅换干净外观（`exp1` vs `exp0`）**：`exp1`（`dc_reg=0.005` + PPISP + `gradient_loss_weight=0`）彻底清除了 `exp0` 中 1F 白墙与檐口带的**粉红/青绿彩斑**以及过曝吹白反差（`1F_win1_left` OOG 由 `42.12%` 降至 `0.31%`），并整体提升了画面色调一致性；**但灰白雾状斑驳与亮白竖向/斜向拉丝从 `10k → 16k → 30k` 照样准时加重**，且三维细长针比例（`50.3%` vs `50.7%`）与暗点（`7.7%` vs `10.8%`）几乎无减缓。
3. **冻结 4k 后 MRNF 与衰减（`exp2` vs `exp1`）**：`exp2` 确实锁死了 4k 后的高斯计数（`1F_win1_left` 全程 `36,363 → 36,477`），并将 `>30 cm` 超大长条从 `494` 砍到 `313`（相比 `exp0` `564` 下降 `44.5%`）、略微降低了叠片率（`5.38x → 5.10x → 4.84x`）。**然而，在相同 `novel_wall_closeup` 与 `novel_high_left` 俯视视角下，`exp2` 从 `3k → 10k → 16k → 30k` 依然发展出几乎同等级别的灰白斑驳与竖条划痕**；2F 上层白墙则在三组中均明显干净得多。

---

### 6.4 关键证伪与根因重定位（连续优化“微百叶 + 竖条”作弊与多视角数据冲突）

本轮隔离实验给出了一个非常明确且关键的**证伪结论**：
- `6.2` 中的机制 1（`ExposureChroma` 过拟合 + `dc_reg = 0`）与机制 3（`gradient_loss_weight = 1.8`）确实主导了 `exp0` 中的**粉红/青绿彩斑与色域发散**（`exp1` 即可完全消除）；
- **但原本假设主导无色竖条纹与细长针的机制 2（4k 后满额 MRNF 无误差盲拆分 + GUT 全局不透明度衰减）被 `exp2` 直接证伪为非主因**——在 `stop_refine = 4000, opacity_decay = 0.0, scale_decay = 0.0`、没有任何新增/拆分高斯的前提下，仅靠 **Adam 连续参数优化（`means_lr / scaling_lr / rotation_lr / opacity_lr / shs_lr`）** 就已足够在 `3k → 30k` 之间将原本健康的白墙彻底“捏”成 `47.6%` 细长针、`69.2%` 极薄片（`s_min` 中位 `0.024 mm`）和 `8.1%` 黑点。

对 [exp2 splat_30000.ply](file:///C:/Users/youwu/Documents/GeosunPre/work/las-norm-ablation/exp2-freeze-topo/splat_30000.ply) 与 [exp2 snap_3000.ply](file:///C:/Users/youwu/Documents/GeosunPre/work/las-norm-ablation/exp2-freeze-topo/snap_3000.ply) 的 `1F_win1_left` 区域进行逐轴三维解剖（另确认 `5..20 cm` 面外区域仅剩 `200` 颗极低不透明度 `0.01` 的残影，全部劣化都发生在 **贴墙 `|off| < 1 cm`** 内），揭示出连续优化造出的**两类共生作弊结构**：

1. **大尺寸竖向半透明薄条带（`s_max > 15 cm`：`3k` `84` 颗 → `30k` `740` 颗）**：
   - 中位尺寸 `26.82 cm × 3.05 cm × 2.48 mm`，主轴几乎纯竖直（`|max_vec · z| = 0.95`），法线紧贴真实墙面（偏角中位 `2.8°`），中位不透明度 `0.28`、亮度 `0.60`。
   - 它们把相邻竖向光照过渡拉伸成整片墙上的长条带，构成了俯视视角里最显眼的**竖向亮白/灰白拉丝**。
2. **倾斜黑白水平微型百叶窗阵列（贴墙微浮雕：`dark < 0.20` `2,944` 颗 + `bright > 0.90` `12,451` 颗）**：
   - 几乎全部留在真实墙面 `|off|` `p10..p90 ∈ [-12.4, +11.0] mm` 内（甚至与墙外偏移无相关性 `corr(off, luma) = -0.055`），但短轴法线相对墙面**猛烈抬头/低头倾斜 `46.7° ~ 51.4°`**（`|min_vec · z|` 中位 `0.44 ~ 0.52`），同时长轴沿走道水平方向拉成细条（`|max_vec · u|` 中位 `0.81 ~ 0.86`）：
     - 极暗条（`luma_p50 = 0.03`）：`8.6 mm × 1.7 mm × 0.69 mm`，不透明度中位 `0.86`；
     - 极亮条（`luma_p50 = 0.99`）：`11.8 mm × 2.1 mm × 0.23 mm`，不透明度中位 `0.69`。
   - **为何这样“作弊”能在训练集上拿高 PSNR，却在高层俯视视角爆炸？** 因为双鱼眼相机采集全程沿 1F 走道行走（`cam_z ∈ [0.70, 1.00] m`），观测 `1F_win1_left` 中心（`z = 1.55 m`）全部为**平视偏仰视（`elev ∈ [-45.5°, -2.5°]`）**，而中庭高处俯视位于**未观测的俯视锥（`elev > 0°`）**。在 `sh_degree = 0`（无视角相关颜色项）时，Adam 利用紧挨墙面的海量高密度切片（`6,080 GS/m²`，是 2F `767 GS/m²` 的 `7.9×`），通过倾斜法线让“亮条/暗条”在不同仰角、方位角下产生**几何遮挡视差伪 BRDF**，去硬拟合多视角光度冲突；一旦进入高处俯视视角，倾斜微百叶彼此间的遮挡比例瞬间翻转，就散裂成一片片灰黑斑驳与白划痕。

进一步回查 `work/小场景-las-norm-out/images/` 投影在 `1F_win1_left` 中心的实拍像素与相机角度，**为什么优化器“必须”在 1F 上如此疯狂作弊、而 2F 不用**：
- **极端的跨视角曝光/溢出与分辨率冲突**：
  - 当背包刚步入昏暗电梯厅回望中庭（`00159-R ~ 00162-R`，`d = 4.83 ~ 6.40 m`），`1F_win1_left` 直接被走廊自动曝光推爆为大面积纯白截断 `(1.00, 1.00, 1.00)`（在 `00164-R` 上先青色通道饱和为 `(0.88, 0.96, 1.00)`）；
  - 同瞬间左右鱼眼（`00168-L` `2.06 m / fov 78.2° / luma 0.937` vs `00168-R` `2.09 m / fov 71.8° / luma 0.758`）在几乎同位置出现明显曝光/亮度跳变；
  - 而沿墙近距离擦墙经过时（`00171-L ~ 00173-L`，`d = 1.03 ~ 1.28 m`），鱼眼清晰拍出了真实墙面的**毫米级厚浆拉毛凹凸颗粒（knockdown stucco）纹理**与背包作业员近距投射阴影（patch 亮度降至 `0.530 ~ 0.615`），导致同一点在不同视角间真实亮度跨越 `[0.53, 1.00]`、单靠全图均匀曝光标量 PPISP 无法吸收局部截断与近距阴影。
- **叠加机制助推**：
  - `normal_flatten_weight = 0.1` 从 `2,400` 步开启后一路将最薄轴压穿到 `0.024 mm`（极薄片在斜视角更容易产生高梯度与遮挡切割）；
  - 基线 `erank_reg = 0.0`：核对 LichtFeld Studio 的 `effective_rank_penalty`（[effective_rank.cuh](file:///C:/Users/youwu/Documents/LichtFeld-Studio/src/training/include/lfs/training/effective_rank.cuh#L16-L56)）可知，当三个尺度均为等大扁平圆盘 `[s, s, eps]` 时有效秩接近 `2.0`，惩罚几乎为 0，只有塌缩为 1D 细长针时 `rank → 1`、`penalty = -log(rank - 0.99999)` 才会急剧抑制拉丝——而在基线中该惩罚被设为 `0.0`；
  - 1F 因前 3k 步“极近距拍出凹凸粒子纹理 + 高误差”，在 3k 之前就被密集拆分到 `6,080 GS/m²`，给了 Adam 比 2F（`767 GS/m²`）多 8 倍的自由度去编织微百叶与竖条。

---

### 6.5 下一步最小验证实验方向

基于上述已实证的机理，后续可直接围绕“**堵死低高视角几何作弊 + 阻断饱和截断像素污染 + 消解近距密片**”安排最小对比验证（固定以 `exp2-freeze-topo` 为 clean 新基线）：

1. **抑制扁平针化与过度压薄（仅动正则化，1 组对比）**：在 `exp2` 基础上开启 `erank_reg > 0`（抑制面内 `s_max / s_mid` 失控拉成 `27 cm` 竖条与 `5:1` 细针，同时先验不伤圆盘贴片）、下调或延后 `normal_flatten_weight`（如 `0.1 → 0.01~0.02`，防止把 `s_min` 压成 `0.02 mm` 锋利叶片参与遮挡微百叶）。
2. **用温和 SH1 吸收真实高光/拉毛明暗微变（替代几何微百叶）**：对比 `sh_degree = 1` + 适度 `sh_rest_reg`，验证少量合法的低阶视角相关反射是否能阻止 Adam 把仰视与平视的亮度差捏进贴墙黑白倾角百叶。
3. **数据源侧降权高光截断与前 3k 近距过密（GeosunPre / 训练权重侧）**：对训练集中单通道/三通道近似饱和像素（例如过曝走廊视图中 wall `>= 0.98~0.99` 的过曝裁剪区）在光子损失中做软掩码/降权，避免 `exp1/exp2` PPISP 仍必须硬拟合已被相机截断成 `255` 的假纯白；并研究在 `0 ~ 3,000` 步降低极近距（`< 1.5 m`）无显著边缘区的高斯初始/早期密度上限（让 1F 向 2F 的 `1,000 ~ 2,000 GS/m²` 靠拢）。

---

### 6.6 `exp3-reg-sh1` 实测结果（正则化阻断微百叶 + 延迟致密化 + 温和 SH1）

在 [exp2-freeze-topo.json](file:///C:/Users/youwu/Documents/GeosunPre/work/las-norm-ablation/configs/exp2-freeze-topo.json) 基础上生成 [exp3-reg-sh1.json](file:///C:/Users/youwu/Documents/GeosunPre/work/las-norm-ablation/configs/exp3-reg-sh1.json)，仅调整以下 7 项参数并完整跑完 `30,000` 步、`3k/10k/16k` 快照转换、30k 全视角 `eval --eval-all`（[exp3 metrics](file:///C:/Users/youwu/Documents/GeosunPre/work/las-norm-ablation/exp3-reg-sh1/eval/metrics_report.txt)）及统一俯视探测渲染（使用 `--sh-degree 1`，并额外补渲染 `--sh-degree 0` 纯 DC 视图以分离 SH1 影响）：
- `erank_reg: 0.0 → 0.008`、`scale_reg: 0.01 → 0.005`、`normal_flatten_weight: 0.1 → 0.01`；
- `sh_degree: 0 → 1`、`sh_rest_reg: 0.0 → 0.001`；
- `start_refine: 0 → 1200`、`stop_refine: 4000 → 4500`（避开 PPISP `warmup = 500`，使模型于 `3,300` 步达到 `3,000,000` 上限、crop 后保留 `2,994,286` 颗）。

#### 6.6.1 30k 全视角 `eval` 与三维审计对比

| 实验组 | 耗时 | `#GS` | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Aligned PSNR / SSIM / LPIPS | Bias (raw) | 法线误差 (deg) | 深度 med / P90 (mm) | `alpha < 0.98` ↓ |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `exp0-baseline` | `843.8s` | `2,984,778` | `24.2966` | `0.8293` | `0.2415` | `25.4524 / 0.8537 / 0.2374` | `(+0.042, +0.041, +0.047)` | `9.0262` | `6.33 / 34.33` | `0.3147` |
| `exp2-freeze-topo` | `804.4s` | `2,994,278` | `24.5354` | `0.8294` | `0.2420` | `25.9264 / 0.8562 / 0.2370` | `(-0.027, -0.026, -0.023)` | **`8.1186`** | `6.38` / **`33.57`** | `0.3198` |
| **`exp3-reg-sh1`** | `873.6s` | `2,994,286` | **`24.7914`** | **`0.8343`** | **`0.2399`** | **`26.2151 / 0.8597 / 0.2353`** | **`(+0.004, +0.001, +0.005)`** | `8.2109` | **`6.15`** / `33.63` | **`0.2681`** |

`exp3-reg-sh1` 3D 白墙审计时间线（`3,000 → 10,000 → 16,000 → 30,000`）：
- **全局**：`OOG: 0.24% → 0.09% → 0.07% → 0.14%` \| `needle >5: 0.2% → 2.1% → 2.4% → 2.6%`（`>10` 仅 `0.6%`） \| `s_min: 7.05 → 3.59 → 2.62 → 2.052 mm` \| `flat<0.1: 3.2% → 14.9% → 22.6% → 30.4%`；
- **`1F_win1_left`**：`n: 25,366 → 29,439 → 29,501 → 29,490`（`4,915 GS/m²`，比 `exp2` `6,080` 降 `19%`） \| `od: 3.81x → 5.29x → 5.99x → 6.11x` \| `>5: 0.0% → 1.2% → 2.7% → 3.4%`（`>10` `0.5%`） \| `>30cm: 21 → 137 → 208 → 249` \| `dark: 0.0% → 1.0% → 4.5% → 8.8%` \| `opac: 0.18 → 0.27 → 0.40 → 0.58` \| `norm_ang: 20.0° → 23.2° → 40.5° → 50.3°`；
- **`1F_between_wins`**：`n: 29,574 → 37,441 → 37,516 → 37,558` \| `>5: 0.1% → 2.6% → 3.3% → 3.4%` \| `>30cm: 0 → 48 → 104 → 153` \| `dark: 2.2% → 3.2% → 6.5% → 11.0%` \| `norm_ang: 9.6° → 27.6° → 43.3° → 49.9°`；
- **`2F_above_clean`**：`n: 4,314 → 4,579 → 4,575 → 4,581` \| `>5: 0.0% → 0.1% → 0.1% → 0.2%` \| `dark: 0.1% → 0.2% → 0.2% → 0.5%` \| `norm_ang: 11.8° → 10.9° → 14.9° → 19.9°`。

#### 6.6.2 视觉核验与机理收敛结论

1. **已打掉的伪影与客观收益**：
   - `erank_reg = 0.008` 与温和 `sh_degree = 1` 将 `1F_win1_left` 细长针由 `47.6%` 暴降至 **`3.4%`**，肉眼对比 `exp2`、`exp3`（`novel_wall_closeup` 与 `novel_high_left`）确认 **1F 墙面与窗框周边密集尖锐的亮白竖向拉丝划痕已显著消除**；
   - `scale_reg = 0.005` 搭配 `erank_reg = 0.008` 不再压暗或打出微孔：`Raw Bias` 从 `-0.027` 几乎归零至 `+0.001 ~ +0.005`，`alpha < 0.98` 由 `0.3198` 降至 **`0.2681`**（4 组最佳），全场景地砖、绿植与全景指标全面创下最优（`PSNR 24.7914` / `Aligned PSNR 26.2151` / `深度中位 6.15 mm`）。
2. **仍未消除、发生形态转移的病灶（为何还留有灰白糊斑）**：
   - **视觉**：`exp3` 在 `1F_win1_left` 下半部与窗间墙面仍呈现从 `3k → 10k → 16k → 30k` 渐进加重的**柔雾状灰白斑驳与宽幅竖带阴影**，且将 `exp3` 以 `--sh-degree 0`（纯 DC）渲染时斑驳形状几乎一致，说明残余糊斑**并非 `SH1` 视角外推所致**。
   - **三维解剖定位**：在未阻断过曝截断帧（`00159-R~00162-R`）的强光度冲突前，Adam 仅将作弊形态从“超薄细针百叶”切换成另外两种贴墙形态：
     1. **加宽的竖向半透明宽条带**（`>15 cm` `477` 颗，中位 `32.21 cm × 8.41 cm × 9.19 mm`、`norm_ang = 1.3°`、`max_vec·z = 0.91`、`opac = 0.48`、`luma = 0.61`）——由于 `erank_reg` 只惩罚长宽比而不限制绝对 `s_max`，宽条带（纵横比 `~3.8 < 5`）成功逃逸了 `erank_reg`；
     2. **更厚更斜的贴墙黑白小圆盘颗粒**（`dark < 0.20` `2,584` 颗、`luma=0.04, opac=0.96, 6.7×2.9×0.92 mm`；`bright > 0.90` `8,955` 颗、`luma=0.99, opac=0.88`）。由于 `normal_flatten_weight` 被调低到 `0.01` 且 `s_min` 被 `erank_reg` 抬厚至 `1.6~2.1 mm`，短轴失去强扁平与法线锚定后反而更自由地歪斜至 **`50.3°`**（`dark` 高达 `69.7°`）、`overdraw` 升至 **`6.11×`**。
3. **结论与下一步（需要推进 `exp4` 数据端过曝掩码对照）**：
   - 单靠训练端正则化 + `SH1` 足以解决**彩斑、细针拉丝与全视角评测指标**，但无法凭空消解训练图里的硬过曝削顶矛盾；只要 `00159-R~00162-R` 这类大面积假 `(255, 255, 255)` 像素还参与光子损失，优化器就会退守到“宽条带 + 高倾角黑白颗粒”里消化冲突。
   - 下一步建议直接做 **`exp4-overexp-mask`**：在 `work/小场景-las-norm-out` 的副本上扩展 `masks/` 自动屏蔽大面积连续高光过曝饱和区（同时把 `normal_flatten_weight` 回调至 `0.03~0.05` 防止短轴倾角飙高，并对 `scale_reg` / 绝对 `s_max` 保持适度约束），验证切断冲突数据源后 `3k → 30k` 的灰白云雾斑驳能否真正收干净。

---

### 6.7 `exp4-overexp-mask` 实测结果（大面积过曝截断掩码 + `normal_flatten_weight = 0.04`）

为在不修改只读原数据集 [小场景-las-norm-out](file:///C:/Users/youwu/Documents/GeosunPre/work/小场景-las-norm-out) 的前提下验证“屏蔽过曝死白截断像素 + 回调扁平约束”是否能清除 `exp3` 残留的柔雾灰白糊斑，构建目录联接副本 [exp4-dataset](file:///C:/Users/youwu/Documents/GeosunPre/work/las-norm-ablation/exp4-dataset) 与基于 `exp3` 仅改动 `normal_flatten_weight: 0.01 → 0.04` 的 [exp4-overexp-mask.json](file:///C:/Users/youwu/Documents/GeosunPre/work/las-norm-ablation/configs/exp4-overexp-mask.json)：

- **高光过曝截断掩码生成规则与统计**：在保留原始 `0`（ignore）与 `128`（segment 天空）掩码带不变的前提下，于 `1/4` 分辨率（`708×708`）检出 `(max(R,G,B) >= 252) & (min(R,G,B) >= 210)` 的区域，经 `5×5` 形态学开运算滤除零星小灯珠/高光点、再用 `7×7` 膨胀覆盖截断过渡晕边，将命中的 `keep (>250)` 像素置 `0`（ignore）：
  - 296 张图共新增屏蔽 `59,269,702 / 1,721,766,546`（总 `keep` 像素的 **`3.44%`**），单图新增屏蔽占比中位数 **`3.29%`**、`p90 = 6.37%`、最高 `23.33%`（`00167-L`）、`00163-R` `19.68%`、`00162-R` `19.21%`、`00160-R` `10.73%`；而 `1 m` 贴墙正常帧 `00172-L` 仅顶部天窗微边屏蔽 `0.62%`，拉毛白墙完整保留（核验后确认 `1F_win1_left` 中心的 `00159-R ~ 00164-R` 走廊过曝帧已被全部置 `ignore`）。

#### 6.7.1 在原始未修改 `masks/` 上的公平 `eval --eval-all` 与三维审计对比

| 实验组 | 耗时 | `#GS` | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Aligned PSNR / SSIM / LPIPS | Bias (raw) | 法线误差 (deg) | 深度 med / P90 (mm) | `alpha < 0.98` ↓ |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `exp0-baseline` | `843.8s` | `2,984,778` | `24.2966` | `0.8293` | `0.2415` | `25.4524 / 0.8537 / 0.2374` | `(+0.042, +0.041, +0.047)` | `9.0262` | `6.33 / 34.33` | `0.3147` |
| `exp3-reg-sh1` | `873.6s` | `2,994,286` | **`24.7914`** | **`0.8343`** | **`0.2399`** | **`26.2151 / 0.8597 / 0.2353`** | **`(+0.004, +0.001, +0.005)`** | **`8.2109`** | **`6.15`** / `33.63` | **`0.2681`** |
| `exp4-overexp-mask` | `855.8s` | `2,996,245` | `23.6354` | `0.8296` | `0.2443` | `24.7180 / 0.8538 / 0.2398` | `(+0.001, -0.000, +0.004)` | `8.2799` | `6.26` / **`32.47`** | `0.2807` |

- **PSNR 下降 `-1.16 dB` 的逐帧核对解释**（[exp4 metrics](file:///C:/Users/youwu/Documents/GeosunPre/work/las-norm-ablation/exp4-overexp-mask/eval/metrics_report.txt) vs [exp3 metrics](file:///C:/Users/youwu/Documents/GeosunPre/work/las-norm-ablation/exp3-reg-sh1/eval/metrics_report.txt)）：`eval` 计分采用**原始未屏蔽过曝区**的 `小场景-las-norm-out/masks/`，最大降幅集中在走廊回望门框爆白帧（`00166-L: -6.55 dB`、`00159-R: -6.27 dB`、`00158-R: -6.26 dB`、`00160-R: -5.71 dB`），而其 `SSIM` 仅变化 `-0.0086 ~ -0.0329`；即便走廊外帧（`00039..00130` & `00169..00186`，`n=230`）均值也从 `24.387` 降至 `23.556 dB`，原因正是室外帧中约 `3%~9%` 日照高光带同样被 `>= 252` 掩码屏蔽训练，评测时回归真实灰白亮度而被按原始死白真值扣分；同时全局深度 `P90` 达到四组最低 `32.47 mm`。

`exp4-overexp-mask` 3D 白墙审计时间线（`3,000 → 10,000 → 16,000 → 30,000`）：
- **全局**：`OOG: 0.22% → 0.09% → 0.08% → 0.14%` \| `needle >5: 0.2% → 2.1% → 2.4% → 2.6%`（`>10` `0.6%`） \| `s_min: 6.19 → 2.74 → 1.98 → 1.493 mm` \| `flat<0.1: 5.1% → 22.8% → 31.9% → 38.9%`；
- **`1F_win1_left`**：`n: 26,029 → 30,853 → 30,872 → 30,879`（`5,147 GS/m²`） \| **`od: 3.44x → 4.10x → 4.44x → 4.74x`**（比 `exp3` `6.11x` 显著降 **`22.4%`**） \| `>5: 0.0% → 1.4% → 2.9% → 3.7%` \| **`>30cm: 6 → 77 → 135 → 196`**（`>15cm` 由 `477` 降至 **`393`**） \| **`dark: 0.0% → 1.5% → 5.0% → 9.1%`**（`2,821` 颗） \| `opac: 0.19 → 0.32 → 0.47 → 0.65` \| **`norm_ang: 21.0° → 25.2° → 41.2° → 49.9°`**；
- **`1F_between_wins`**：`n: 29,541 → 37,460 → 37,536 → 37,547` \| `od: 4.00x → 3.54x → 3.43x → 3.52x` \| `>5: 0.1% → 2.8% → 3.5% → 3.6%` \| `>30cm: 0 → 46 → 93 → 138`（比 `exp3` `153` 降） \| `dark: 2.3% → 3.1% → 5.8% → 9.9%`（较 `exp3` `11.0%` 微降） \| `norm_ang: 9.1° → 25.3° → 40.8° → 48.0°`；
- **`2F_above_clean`**：`n: 3,447 → 3,645 → 3,638 → 3,632`（`605 GS/m²`） \| `od: 2.07x → 3.33x → 3.36x → 3.37x` \| `>30cm: 1 → 11 → 19 → 23` \| `norm_ang: 10.5° → 9.2° → 12.8° → 16.7°`。

#### 6.7.2 第二次关键证伪与“真·最后一公里”机理定位

在俯视 `novel_wall_closeup` 与 `novel_high_left` 渲染对比中，**`exp4` 虽然把 1F 叠片过绘压低了 `22%`（`6.11× → 4.74×`）、大竖条减至 `196` 个、2F 密度/过绘显著更省，但在 `1F_win1_left` 与窗间墙面上，从 `3k → 10k → 16k → 30k` 依然长出了与 `exp3` 几乎一致的柔雾灰白斑驳**，`30k` 时甚至保持了 **`9.1%` 黑点（`2,821` 颗，`luma=0.03, opac=0.97, norm_ang=67.3°`）、`9,009` 颗亮白点（`luma=0.99, opac=0.89, norm_ang=60.8°`）与 `49.9°` 墙面短轴倾角**。

`exp4` 对剩余“柔雾灰白糊斑 + 倾角黑/白颗粒”完成了最后一层隔离，暴露出**单靠屏蔽 `>= 252` 硬过曝依然消不掉的本质瓶颈**：
1. **未达 `252` 阈值的中等/近过曝跨视角亮度与微表面尺度差仍在**：屏蔽 `00159-R~00164-R` 后，重查 `exp4-dataset` 上 `1F_win1_left` 仍保留的两组近/远帧：
   - 室外远眺帧（`00039..00130`, `n=35`, `d=8.55~14.96 m`）：墙面亮度平稳集中在 `luma p10/p50/p90 = 0.556 / 0.645 / 0.719`（`std = 0.095`）；
   - 贴墙出廊与近擦段（`00165..00186`, `n=10`, `d=1.03~3.36 m`）：从走廊刚步出但未触发 `>=252` 掩码的 `00165-R ~ 00168-L` 仍亮至 **`0.818 ~ 0.937`**（且 `00168-L/R` 同瞬差 `0.937` vs `0.758`），紧接 `1.03~1.28 m` 近贴墙拍出**毫米级拉毛灰泥颗粒（`1.09 mm/px`）+ 作业员身体投射暗影**时又跌至 **`0.530 ~ 0.615`**；而 2F 最近距离仅 `p50 = 2.79 m`（`2.83 mm/px`），因此 2F 既没拍出近距颗粒冲突，初期也没被拆到 1F 的 `8.5×` 密度（`605` vs `5,147 GS/m²`）。
2. **渲染法线损失对“多层单高斯倾角抵消”完全盲区（决定性几何漏洞）**：对 `exp4 @ 30k` 做短轴矢量定向统计揭示了一个惊人规律——`ALL` 中 **`67.4%`**、`dark/bright` 中 **`76.0% / 75.2%` 的高斯最短轴刻意朝 `+z`（正上方）剧烈仰倾**（`mean_mv_z = +0.37 ~ +0.39`）：
   - 在全部处于低处（`z = 0.70~1.00 m`、视角仰倾）的训练相机里，`4.74` 层叠加高斯的表面法线在 alpha 混合后仍可抵消为正确平均法线（所以全视角渲染法线误差只有 `8.28°`，`normal_loss` 与 `normal_consistency` 均**无法阻止单高斯自转到 `50°~67°`**）；
   - 这使优化器能把上万颗 `0.68~1.13 cm`、厚 `1.4 mm` 的黑/白小圆盘在低处仰视锥内做成**斜向视差瓦片**来分别迎合 `0.53` 与 `0.94` 的亮度差；一旦进入未经训练观测的高处俯视视角（`+z` 朝下看），这些朝 `+z` 抬头的黑/白瓦片正好**正面迎向俯视镜头**，从而完整暴露为灰黑斑点与云雾糊斑。
3. **`erank_reg` 只限长宽比、放行宽条带**：`exp4` 中 `1F_win1_left` 仍有 `393` 颗 `>15 cm`、`196` 颗 `>30 cm` 的宽竖条（中位 `29.73 cm × 7.24 cm`、`norm_ang = 1.5°`、`|max_vec·z| = 0.93`、`opac = 0.35`）。

#### 6.7.3 工程落地判断与下一步推荐

1. **推荐优先落地的配置基线**：在不动训练器代码的前提下，**`exp3-reg-sh1` 是五组实验中的综合最优甜点**——完全消除粉红/青绿彩斑与尖锐细丝划痕，`PSNR (24.7914)` / `Aligned PSNR (26.2151)` / `SSIM (0.8343)` / `LPIPS (0.2399)` / `alpha<0.98 (0.2681)` / 深度中位误差（`6.15 mm`）五组最优且 `Raw Bias (+0.001~+0.005)` 基本归零（实际工程落地时可在 `exp3` 上把 `normal_flatten_weight` 取 `0.03~0.04` 以压低 `overdraw`）。
2. **暂不建议直接将 `>=252` 全图硬过曝掩码合并进 GeosunPre 默认 `masks/`**：`exp4` 证明对日照白墙也统一硬裁 `>=252` 会无谓牺牲正常视角亮度校准与 PSNR，却挡不住 `00165~00168` 准过曝和 `1 m` 拉毛阴影引发的倾角小圆盘作弊。
3. **如需彻底消灭最后的 1F 柔雾灰白糊斑，必须从“单高斯防自转 + 绝对尺度封顶 + 近距防过密”三条底层约束突破**（需改动 LichtFeld / GeosunPre 策略）：
   - **单高斯法线/旋转锚定正则**：在优化器直接约束每颗扁平高斯的 `min_vec` 与其局部渲染/先验墙面法线一致（打破多层正负/斜向倾角在 pixel-wise blended normal 下互相蒙混的漏洞，直接封死 `50°~67°` 倾斜瓦片）；
   - **绝对最大尺度软上限**：对平坦弱纹理区的高斯施加 `s_max` 软上限（比如超过 `5~8 cm` 才生效的 hinge penalty），精准卡死 `30 cm × 7 cm` 宽条带而不影响小圆盘导致 `alpha` 微孔；
   - **前 3k 近距弱纹理限密**：在 `< 1.8 m` 且无显著图像/几何边缘区域限制 early MRNF 拆分速率，避免 1F 长出 2F `7~8×` 的冗余自由度。
