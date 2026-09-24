# 篮球场结构角点与原生鱼眼轨迹核查

这一轮沿[球场 L/R 轨迹平面核查](basketball_sfm_court_trajectory_probe.md)继续找可辨认的近地面结构点，目的在于区分“形状相同”“现有 SfM 已覆盖”和“确实缺少可信轨迹”。输入 pose 数值与 LAS 不参与角点识别；两台鱼眼的曝光仍视为异步，每张图保留独立位姿。文中的几何检查只把现有等权**视觉 SfM 解及其拟合候选内参**当作待检验对象，不当作真值。本轮没有改 COLMAP/LFS 训练输入，也没有执行新的 BA。见[三色角点协议](../work/basketball/global_sfm/sfm_court_anchor_probe/protocol.json)、[检测与配对结果](../work/basketball/global_sfm/sfm_court_anchor_probe/results.json)、[原图覆盖核对 v2](../work/basketball/global_sfm/sfm_court_anchor_probe/coverage_v2.json)。初版 `coverage.json` 曾把 COLMAP 的 `UINT64_MAX` 无效点号经 NumPy 转为浮点，导致已建图点邻近度错误；以 v2 为准。

在 138 张按球场可见区选出的临时朝下视图里，27 张有 HSV 红/绿/蓝三色交点候选，17 条已有图像单应边能关联这些候选，只有 4 组构成连续三帧。[四组视图](../work/basketball/global_sfm/sfm_court_anchor_probe/review/junction_triplets.jpg)和[对应原生鱼眼裁剪](../work/basketball/global_sfm/sfm_court_anchor_probe/review/junction_native_crops.jpg)可直接检查检测位置。前两张由现有视觉位姿三角化、第三张留出的结果如下；误差单位分别为 **640×640 临时视图像素**和**原生鱼眼像素**。此处原图坐标由临时视图检测点经候选内参映回，并非在原图重新精定位；后面的 12 图实验才直接在原图定位。

| 连续三帧 | 第三帧视图误差 | 第三帧原图误差 | 原有模型在每张交点 16 原图像素内的共同 3D 轨迹 | 判断 |
| --- | ---: | ---: | ---: | --- |
| L50–52 | 1.19 | 2.52 | 3 | 结构稳定，但原前端已覆盖 |
| L103–105 | 4.94 | 11.76 | 1 | 角点定位或几何不够稳 |
| L233–235 | 1.21 | 2.28 | 0 | 原生图清楚，值得延长轨迹；首点在临时视图边界 1.5 px |
| R223–225 | 1.28 | 2.38 | 1 | 结构稳定，但原前端已覆盖 |

L210–L289 提醒我们不能只凭场地色块拓扑连远距离回访：三色区域交集的**质心**经 LoMa 图像单应传递只差 0.12 个视图像素，但按色块边缘重新定位的角点差 1.30 像素。120 像素局部高通纹理相关约 0.28，邻近 L209–L210 约 0.58；这些相关数受视角和照明影响，不能直接作真假阈值。[原生图场景上下文](../work/basketball/global_sfm/sfm_court_anchor_probe/review/L209_L210_L289_native.jpg)也没有给出可确认的共同独特纹理。旧模型里两图没有共同 3D 轨迹，先前的极线/平面大残差仍不能被简单解读为 pose 漂移。这条边保持未证实，**不注入 BA**。

L233–235 在临时朝下视图的边缘，但在原生鱼眼图中是清楚的球场红色区域左角。于是继续在 **L231–L242 连续 12 次曝光**中，按图像内容给出两个角点的粗搜索框，再在原图局部由三色区域到各自边界的距离定位交点。粗框只是图像选区，既非精确人工像素标注，也不使用 pose 引导关联。[原生角点协议](../work/basketball/global_sfm/sfm_court_native_anchor_probe_v2/protocol.json)、[完整逐帧结果](../work/basketball/global_sfm/sfm_court_native_anchor_probe_v2/results.json)、[左角逐帧裁剪](../work/basketball/global_sfm/sfm_court_native_anchor_probe_v2/review/paint_left_corner_native_crops.jpg)、[右角逐帧裁剪](../work/basketball/global_sfm/sfm_court_native_anchor_probe_v2/review/paint_right_corner_native_crops.jpg)均已冻结。最初搜索半径 60 原图像素使一个右角落在搜索框边上、运行未完成；v2 扩到 80 像素后才产出下列结果。

左角在 12 张图中都位于候选 179° 鱼眼有效域内，各帧距最近**已建图**特征 **42–98 原图像素**，16 像素内均没有现有 3D 轨迹；这是原前端遗漏的近地面结构点。[逐帧有效域与掩码核对](../work/basketball/global_sfm/sfm_court_native_anchor_probe_v2/eligibility.json)还显示：两侧角点在全部 12 张图中的掩码值均为 255、拟合候选相机 160° 域内有效，而旧模型每张恰有 3072 个特征点。因此空白**不是**由已保存的掩码或 160° 域直接剔除造成；尚不能区分 LoMa 检测、排序和特征数上限各自的贡献。用旧视觉 pose 和候选内参，仅由 L233、L239 的两个原生鱼眼射线三角化，另外 10 张的预测误差中位数 **1.80 px**、最大 **3.57 px**；换成 L231、L242 作两张支持图，中位数 **4.12 px**、最大 **7.26 px**。12 张共同拟合一个 3D 点的残差中位数 **1.27 px**、最大 **3.49 px**。不同两图支持组合并非完全等价，说明这还不是零误差的像素真值；这些组合也是探索后选择的，不能称作盲测。右角同样缺少贯穿 12 张的旧 3D 轨迹，但上述两个支持组合的留出中位数分别为 **5.47、4.05 px**，12 图拟合中位数 **3.57 px**；白线边缘定位、内参和位姿都可能贡献误差，暂不作为高权重约束。[两轨状态](../work/basketball/global_sfm/sfm_court_native_anchor_probe_v2/DIAGNOSTIC_ONLY.json)。

左角提供了一个重要的前端改进方向：沿原生鱼眼图寻找有明确物理身份的场地结构，可补上只取 3072 个 LoMa 特征留下的近地面空白。但现有视觉 pose 已能很好预测这个短区段的左角；单靠这一条 12 视图轨迹不足以修正全场位姿，更不能解释全部训练视角的 3DGS 不一致。下一步应在更多 L/R 区段、不同深度的静态结构上复用原图识别和跨视图核对，保留一部分可靠角点作为**不参加 BA 的验证点**。只有积累足够分布广、非重复纹理且定位稳定的训练轨迹，再做隔离的异步逐图 pose/共享候选鱼眼内参优化，并用原图留出重投影与同一批 LFS 训练视角比较是否真的改善。

后续已执行[全场原生图双角扫描](basketball_sfm_court_paint_pair_probe.md)：它找到另外几个未被旧模型覆盖的 L 近地面局部轨迹，也暴露了自动选点接上篮架和 R 区段误差较大的问题，均仍处于诊断阶段。

复算入口：

```bash
uv run --no-sync geosunpre sfm-court-anchor-probe --work work/basketball
uv run --no-sync geosunpre sfm-court-anchor-coverage --work work/basketball
uv run --no-sync geosunpre sfm-court-native-anchor-probe --work work/basketball
uv run --no-sync geosunpre sfm-court-native-anchor-eligibility --work work/basketball
```
