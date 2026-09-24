# 篮球场 L/R 轨迹的球场平面一致性诊断

这轮把此前仅覆盖 L120–122、L150–152 的球场平面实验扩展到有 pose 记录的 L/R 轨迹。目的是找出**有图像证据支持**、值得单独优化 SfM 位姿的区段；目前没有改动 SfM 模型或 LFS 的 COLMAP 训练输入。[状态标记](../work/basketball/global_sfm/sfm_court_trajectory_probe/DIAGNOSTIC_ONLY.json)。

## 取样与几何定义

只扫描全场清单中的 536 张 L/R 图像。原始 pose 记录仅用于已有清单的图像纳入资格，其数值与 LAS 均不参与取样、匹配或单应拟合。四分之一原图分辨率的 HSV 红/绿/蓝连通区域用于挑选球场可见的相机内连续三帧；这是粗提案，并非球场分割真值。每 10 帧最多取每个相机的一组三帧，得到 L 27 组、R 19 组，共 138 张图像。R 在 43–113 的多个十帧段没有达到颜色面积门槛；[原图接触表](../work/basketball/global_sfm/sfm_court_trajectory_probe/review/court_visibility.jpg)显示这些视角不少朝向绿化或球场边缘。[完整取样结果](../work/basketball/global_sfm/sfm_court_trajectory_probe/scan.json)保留了 536 张的逐图统计。

每张选中图用**现有视觉解中拟合的候选鱼眼内参**生成 640×640、80°、朝下 45° 的临时视图。现有原生鱼眼 LoMa 匹配映射到该视图，只保留得分 ≥0.9、两端均处于同类球场颜色区域的候选，再用 MAGSAC 拟合平面单应。候选配对来自原有前端的时间/外观图；即使是同编号 L/R，也必须有实际匹配才能通过。每张曝光仍有独立位姿，临时视图不会变成新的 SfM 相机。**视觉 pose 只在图像单应完成后**用于最佳单平面兼容性检查，其数值也不是标定真值。协议和结果分别见[扫描协议](../work/basketball/global_sfm/sfm_court_trajectory_probe/scan_protocol.json)、[配对协议](../work/basketball/global_sfm/sfm_court_trajectory_probe/pair_protocol.json)、[逐边结果](../work/basketball/global_sfm/sfm_court_trajectory_probe/pairs.json)。

## 主要结果

在 138 张图涉及的 694 对前端候选中，113 对有至少 20 条球场高分匹配并拟合出单应；110 对的覆盖范围还足以运行现有视觉 pose 加最佳单平面检查。101 对相距不超过两帧的边全部在现有稀疏模型中有共同 3D 轨迹。它们的逐边最佳平面与图像单应的网格位移差中位数之中位数为 **0.69 个 640 视图像素**，最大 **2.03 像素**；这些边上同色的原生高分匹配相对候选视觉 pose 的极线角误差中位数之中位数约 **0.065°**。[稀疏轨迹与极线佐证](../work/basketball/global_sfm/sfm_court_trajectory_probe/corroboration.json)。这说明已覆盖的短区段没有显著的球场平面位姿冲突，**不说明**全场照片或 3DGS 训练视角已经对齐。

15 组三条边都可拟合的三帧中，L156–158 与 L230–232 的单应闭环网格位移差中位数分别为 **2.96、4.46 个视图像素**，高于其余组。两组跨两帧边分别只有 34、43 个单应内点，图像主要是重复绿色铺装；相邻边点更多，而旧稀疏模型分别有 477、276 条对应的共同 3D 轨迹。它们提示跨帧平面拟合会被重复纹理影响，目前不足以把问题归因于 pose。[异常与对照叠加图](../work/basketball/global_sfm/sfm_court_trajectory_probe/review/pair_overlays.jpg)从左到右是源视图、目标视图、单应叠加。

最极端的 L210–L289 是**单条远距离外观候选**：24/24 个入点足以给出平滑单应，颜色边界看起来也相近，但原生高分匹配全部落在球场色块上，两图在保存的稀疏模型中共同 3D 轨迹数为 **0**，相对于该视觉 pose 的极线角误差中位数约 **20.04°**，最佳单平面仍差 **195.48 个 640 视图像素**。这更像无法独立确认的重复铺装候选，不能将其大残差直接解释成全场 pose 漂移，更不能将其作为 BA 约束。相反，L169–R247 这类实际重访匹配有 197 个平面内点、492 条共同 3D 轨迹，最佳平面差约 1.04 像素；跨 L/R 曝光并不需要假设同步。

L169–R249 另有 33 个内点、558 条共同 3D 轨迹和约 **4.34 像素**的平面差，值得继续看，但它与相近回访 L169–R247/248 的结果不一致，且单应主要由绿色区域支持。为避免只看同镜头三帧，又对所有拟合边组成的 **25 个图像单应三角形**做了不使用 pose 的闭环检查。[跨 L/R 的完整闭环结果](../work/basketball/global_sfm/sfm_court_trajectory_probe/graph_cycles.json)中，L169→R249 的直接单应分别与经 R247、R248 中转的单应相差 **4.64、4.27 像素**（中位数）；L169→R247→R248 的闭环差仅 **1.30 像素**，R247–249 本地闭环为 **0.60 像素**。因此 L169–R249 的 4.34 像素 pose 检查异常至少部分反映了它自己的图像单应不稳定，不能据此单边归因于位姿。L231–233 也出现 **3.27 像素**的闭环差，延续 L230 附近的稀疏长边问题。现有证据尚不能拆开匹配点、非严格平面、候选内参与位姿的影响。此处不基于单边结果启动局部 BA。

## 对后续 SfM 的约束

本实验复用的 LoMa 点仍受原生前端 160° 域和设备掩码限制；前面的 179°/朝下视图实验也未提供可确认的新近地面同名点。因此这批平面诊断主要覆盖**已有前端看得到**的球场区域，不能验证被截掉的画面下缘。红/绿/蓝区域同时参与点筛选，后续颜色 IoU 仅用未拟合的区域像素检查整体对齐，不能称为完全独立真值。逐边还允许各自拟合最佳平面，局部低误差不能证明整条轨迹共用一个正确场地平面。

目前优先检查有独立身份信息的球场白线、颜色交点及邻近静态立体结构，特别是 L156、L230 和 L169–R249 的局部区域；再把能跨帧重复确认的原生像素对应纳入**隔离的**位姿/内参候选，与同一训练图像的对齐误差比较。不要提高普通重复铺装点的权重。现有 LFS 训练视角一致性问题仍未解决，本轮结果不能代替对全图、非平面结构及最终渲染的检查。

已另做[密集图像单应后处理的隔离试验](basketball_sfm_court_dense_probe.md)：它改善了上述少数大闭环，但也在球架等非平面内容上严重失手，因此没有晋升为新的 SfM 约束。

复算入口：

```bash
uv run --no-sync geosunpre sfm-court-trajectory-probe --work work/basketball --stage scan
uv run --no-sync geosunpre sfm-court-trajectory-probe --work work/basketball --stage pairs
uv run --no-sync geosunpre sfm-court-trajectory-probe --work work/basketball --stage corroborate
uv run --no-sync geosunpre sfm-court-trajectory-probe --work work/basketball --stage cycles
uv run --no-sync geosunpre sfm-court-trajectory-probe --work work/basketball --stage review
```
