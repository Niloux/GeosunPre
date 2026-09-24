# 篮球场平面密集配准的隔离试验

此前沿 L/R 轨迹的[图像单应诊断](basketball_sfm_court_trajectory_probe.md)显示，少数仅有 30–40 个内点的跨帧边使单应闭环偏差较大。此处试验一个受控的后处理：以那些 LoMa/MAGSAC 单应为初值，在 **640×640 临时朝下视图**上用 OpenCV ECC 的灰度单应模型做密集图像配准，再检查多视图闭环与现有视觉 SfM pose 的兼容性。ECC 不读入 pose、导航数值或 LAS；视觉 pose 只在图像配准后比较。既没有改动原生鱼眼匹配点，也没有输出新的 COLMAP/LFS 输入。[协议](../work/basketball/global_sfm/sfm_court_dense_probe/protocol.json)、[逐边与闭环结果](../work/basketball/global_sfm/sfm_court_dense_probe/results.json)、[诊断状态](../work/basketball/global_sfm/sfm_court_dense_probe/DIAGNOSTIC_ONLY.json)。

纳入 110 条原单应有至少 20 个内点的边，108 条完成数值配准，2 条失败。24 个原有图像单应三角形可完整比较，17 个闭环的网格位移差中位数下降；全部可比闭环的该指标中位数从 **0.95→0.92 个 640 视图像素**，总体改善很小。原本偏差最大的 L169–R247–R249、L230–232、L169–R248–R249、L231–233、L156–158 分别从 **4.64→1.48、4.46→1.43、4.27→1.60、3.27→1.20、2.96→0.99 像素**。L169–R249 的直接单应与现有视觉 pose 加最佳平面的差也从 **4.34→1.69 像素**，其蓝色区域重叠 IoU 从约 **0.48→0.85**。[源图、目标图、LoMa 单应叠加、密集单应叠加的对照](../work/basketball/global_sfm/sfm_court_dense_probe/review/selected_overlays.jpg)能看到一些边界对齐改善。这支持“少量稀疏格纹点使直接单应不稳”的解释，但**不是**新的点身份或正确 pose 的证明。

密集配准同样会明显失手：5 条完成数值配准的边在共同可见的球场网格上移动超过 10 像素。R188–R189 的移动中位数约 **62.1 像素**，ECC 系数只有 **0.23**，相对于原视觉 pose 的最佳平面差从 **1.72 恶化到 34.84 像素**；其临时视图里篮球架、篮板等非平面内容占了大量近似球场颜色区域。L63–65 等边也发生 20 像素以上的移动。R150–L289 一边配准未收敛，L37–L39 一边配准后共同球场网格不足。即便在 97 条相距不超过两帧且均有前后 pose 检查的边上，最佳平面差中位数之中位数也从约 **0.69 变为 0.78 像素**；46 条改善，51 条恶化。不能以“ECC 返回成功”决定整段前端替换。

当前候选内参仍来自旧视觉解，临时针孔视图只是采样/诊断坐标；原生图像下缘仍受旧前端 160° 域与设备掩码限制。ECC 优化的就是这些训练图像的灰度，可能对重复格纹、光照及非平面目标过拟合。颜色 IoU 与三角闭环能排除一些坏解，却不能给出物理同名点真值。下一步只宜在可确认的白线交点、颜色拐角或邻近静态立体结构上审核原生像素对应，并以隔离的 SfM 候选验证是否真的改善同一批训练图像；本轮结果不晋升为 BA 约束或 pose 更新。

复算入口：

```bash
uv run --no-sync geosunpre sfm-court-dense-probe --work work/basketball --stage probe
uv run --no-sync geosunpre sfm-court-dense-probe --work work/basketball --stage review
```
