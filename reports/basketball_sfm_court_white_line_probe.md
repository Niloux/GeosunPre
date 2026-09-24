# 篮球场白线交会与旧 SfM 轨迹核查

这轮从[全场双角提案](basketball_sfm_court_paint_pair_probe.md)的原生 L/R 图像出发，把 HSV 三色交点仅用作 256×256 像素搜索中心，再独立拟合球场白线。输入仍是 536 张有对应 pose *记录*的 L/R 异步曝光图像；M、无对应记录的图像、导航 pose 数值和 LAS 均未参与角点选择。已有等权视觉 SfM 的 pose 与拟合鱼眼内参只用于**定位完成后的诊断**，不是标定或位姿真值。本轮没有新增 BA 约束，没有修改 COLMAP/LFS 训练输入。[冻结协议](../work/basketball/global_sfm/sfm_court_white_line_probe/protocol.json)、[逐帧结果](../work/basketball/global_sfm/sfm_court_white_line_probe/results.json)、[图像诊断状态](../work/basketball/global_sfm/sfm_court_white_line_probe/DIAGNOSTIC_ONLY.json)。

在原图窗口里先取低饱和、高亮度白色像素，再分别要求它贴近红/绿、红/蓝、绿/蓝色块。每组像素拟合一条线，求红/绿线与蓝侧线的交会。拟合要求至少 35 个线内像素、40 px 线段跨度、15° 交角；不同有效交会相差不超过 8 px，线段不能离 HSV 搜索中心或交会过远。这些只是**图像几何筛选**，不能证明跨帧物理身份，尤其白漆有宽度时不存在唯一的“零宽”交点。[全部 12 段逐帧原图裁剪](../work/basketball/global_sfm/sfm_court_white_line_probe/review/)中红十字是旧 HSV 点、黄十字是白线交会，彩色细线是拟合的边界。

| 段与角点 | 图像几何通过 | 同帧旧 HSV 留出中位 | 白线留出中位 / 最大 | 说明 |
| --- | ---: | ---: | ---: | --- |
| L125–129 左 | 5/5 | 2.85 px | 1.73 / 2.57 px | 只有短轨迹，不能单独支撑 pose 更新 |
| L233–243 左 | 11/11 | 1.91 px | 1.30 / 4.99 px | 9 张共同留出帧；最大误差比 HSV 的 3.23 px 更高 |
| L252–259 左 | 7/8 | 1.68 px | 0.71 / 3.64 px | 5 张共同留出帧 |
| R267–281 左 | 15/15 | 5.72 px | 1.44 / 6.61 px | 12 张共同留出帧；R281 白线点越过当前候选 160° FOV，未纳入留出 |
| R267–281 右 | 13/15 | 12.69 px | 8.12 / 19.20 px | R267–268 需外推过远而拒绝；固定支撑帧 R277 不稳定 |

留出统一沿用原双角审计已冻结的约 1/4、3/4 帧，只在**两种定位都有效**的同一批非支撑帧上对比，均为原图像素。它衡量这些观测对保存视觉模型的兼容性，不能把误差直接归因于 pose。R 左角颜色点位曾比白线交会偏 24 px，故此前 R 左角的 6.31/20.10 px 全段 HSV 留出很大一部分是定位问题。L233 左角的白线方案虽有更低中位数，却没有改善最坏帧，也不足以证明全局图像一致性已经解决。[R267–281 白线逐帧第 1 页](../work/basketball/global_sfm/sfm_court_white_line_probe/review/R267_281_page_01.jpg)、[第 2 页](../work/basketball/global_sfm/sfm_court_white_line_probe/review/R267_281_page_02.jpg)、[第 3 页](../work/basketball/global_sfm/sfm_court_white_line_probe/review/R267_281_page_03.jpg)；[L233–243 第 1 页](../work/basketball/global_sfm/sfm_court_white_line_probe/review/L233_243_page_01.jpg)。

[图像阈值扰动](../work/basketball/global_sfm/sfm_court_white_line_probe/stability.json)把白色饱和度/亮度阈值与颜色邻接半径分别改变，完全不使用 pose。L233 左角各组可比帧最大移动约 0.31–1.98 px；R 左角为 0.87–3.69 px，表明线的*检测*相对稳定。R 右角在严格白色阈值下最大移动 7.46 px，放宽白色阈值后有两帧不再通过几何门槛。R277 右角本身的不同白色阈值定位差可到 7.46 px，其多线交会差也达到 7.6 px，接近 8 px 门槛。[R 右角白线支持像素](../work/basketball/global_sfm/sfm_court_white_line_probe/review/R270_281_right_pair_masks.jpg)显示：R277–280 的白漆宽带与斜浅交角会把红/蓝侧和绿/蓝侧的交会拉开，不能把一个数学线交会当成唯一可靠的物理点。

白线新位置也不能一概视作“LoMa 未覆盖”。[旧模型邻近特征与 3D 点投影](../work/basketball/global_sfm/sfm_court_white_line_probe/coverage.json)显示，L233–243 左角在 11 帧中仍有 **0/11** 帧找到 16 px 内旧 3D 特征；R267–281 左角为 **6/15**，右角为 **2/13**。R 左角最近的旧点 `27882` 在 R275–279、R281 已有观测；R 右角最近的旧点 `203485` 在 R267–270 已有观测。这些点只是空间邻近，未被当作严格的同一白漆几何点。把各点投到本段**没有该点观测**的曝光，R 左角的白线差中位/最大为 4.81/9.55 px（9 帧），R 右角为 3.01/9.39 px（11 帧）。右角这项独立于 R277 两视图支撑的检查，比 8.12 px 中位小得多；它反过来说明后一个数字受角点定义和支撑帧影响，尚不能证明 R pose 漂移。[右角旧点投影叠加第 1 页](../work/basketball/global_sfm/sfm_court_white_line_probe/review/R267_281_model_right_01.jpg)、[第 2 页](../work/basketball/global_sfm/sfm_court_white_line_probe/review/R267_281_model_right_02.jpg)、[第 3 页](../work/basketball/global_sfm/sfm_court_white_line_probe/review/R267_281_model_right_03.jpg)中青十字是旧 3D 点投影；[左角第 3 页](../work/basketball/global_sfm/sfm_court_white_line_probe/review/R267_281_model_left_03.jpg)可见晚段仍有数像素差。

当前唯一较扎实的**新近地面轨迹候选**是 L233–243 左角：11 帧可见、阈值扰动小、旧模型没有近邻 3D 点，保存视觉模型对它的留出中位约 1.3 px。不过原模型已经能较好解释它，单独加权难以修复全场训练视角一致性。R 双角则不宜直接高权重注入：左角一部分已被旧 3D 点覆盖，右角浅交角和白漆宽度使坐标定义不稳。后续若继续动 SfM，需先找 R277–281 同一批曝光里**独立于球场白漆、不同深度**的可靠静态轨迹，并与已冻结的地面角点共同检查；只有两类观测支持同方向的修正，才值得做隔离的逐曝光位姿/鱼眼内参优化，再用全部 536 张训练图检查 LFS 的训练视角一致性。

复算入口：

```bash
.venv/bin/python -m geosunpre sfm-court-white-line-probe --work work/basketball --stage all
```
