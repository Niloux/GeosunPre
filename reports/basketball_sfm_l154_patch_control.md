# L154 白斑与静态结构的同支撑帧留出核查

围绕 L153–155 球场红绿边界旁的白色磨损短划，先对五张**原生鱼眼图** L151–155 做一次有界 LoMa 裁片采样，再从原图选择附近地面候选和不同深度的固定结构。所有曝光按独立相机位姿处理。输入导航 pose 数值、LAS、M、无 pose 记录的图像均未参与；保存的等权视觉 SfM 位姿和 L 鱼眼内参只用于选择冻结之后的留出诊断，均不当真值。本轮没有修改 BA、COLMAP/LFS 输入，也没有划分测试图。

## 图像侧候选

以白斑为粗中心截取五个 768×768 原图裁片，每图检测 1536 个 LoMa 特征，匹配全部 10 对图像。[采样协议](../work/basketball/global_sfm/sfm_l150_crop_loma_probe/protocol.json)、[完整匹配统计与候选坐标](../work/basketball/global_sfm/sfm_l150_crop_loma_probe/results.json)。白斑附近有 **44** 条五帧相邻匹配链，其中 **9** 条满足所有间隔不超过两帧的匹配闭环，**6** 条满足十边全闭环。原图审阅后，重复红砖纹上的高分十边闭环仍被否决；只有白斑边缘两条保留为**候选**，它们相距很近且白漆磨损，不具备精确独立地面控制点的资格。[冻结选择](../work/basketball/global_sfm/sfm_l150_crop_loma_probe/selection.json)、[五帧原图裁片](../work/basketball/global_sfm/sfm_l150_crop_loma_probe/review/frozen_selected_tracks.jpg)、[首轮留出审计](../work/basketball/global_sfm/sfm_l150_crop_loma_probe/audit.json)。

为核查“整张 L154 曝光统一偏移”，又从现有原生 LoMa 匹配的 L153–155 三边严格闭环中，按原图固定构件选择了 **6** 条轨迹。它们在保存模型中三帧均未建为 3D 点；但保存的相机位姿曾由其它特征优化，因此这里只是留出点位，不是独立相机真值。[选择协议](../work/basketball/global_sfm/sfm_l154_static_control_probe/selection_protocol.json)、[逐点坐标和匹配分](../work/basketball/global_sfm/sfm_l154_static_control_probe/selection.json)、[静态构件原图裁片](../work/basketball/global_sfm/sfm_l154_static_control_probe/review/frozen_static_controls.jpg)。

## 相同 L153/L155 支撑下的 L154 误差

每条轨迹都用 L153 和 L155 的冻结像素射线三角化，**只在 L154 预测**。下表的有符号残差是“视觉模型预测像素 − 原图匹配像素”；像素数均为原生 3000×4000 图像坐标。[静态结构逐点审计](../work/basketball/global_sfm/sfm_l154_static_control_probe/audit.json)、[白斑同支撑比较](../work/basketball/global_sfm/sfm_l154_static_control_probe/comparison.json)。

| 图像侧轨迹 | L153/L155 射线夹角 | L154 残差 (x, y) | 误差 |
|---|---:|---:|---:|
| 近处黑色立柱脚 | 5.45° | (−0.08, −0.37) px | 0.38 px |
| 近处灰色立柱脚 | 8.21° | (−0.68, +0.59) px | 0.90 px |
| 塔楼外轮廓带 | 2.71° | (−2.15, +0.13) px | 2.15 px |
| 塔楼立面接缝 | 2.48° | (+1.51, −0.01) px | 1.51 px |
| 连廊屋顶转角 | 1.89° | (−0.60, −0.53) px | 0.80 px |
| 右侧立面格线 | 1.12° | (−2.19, −3.75) px | 4.34 px |
| 白斑边缘候选 A | 49.17° | (+1.83, +4.67) px | 5.02 px |
| 白斑边缘候选 B | 46.77° | (+1.65, +4.80) px | 5.08 px |
| 红绿边界上下文点 | 36.56° | (+2.02, +4.14) px | 4.61 px |

右侧立面格线虽有较大误差，但其射线夹角只有 **1.12°**，三角化更不稳定，且残差方向与白斑相反；不能把它并入白斑一致偏移。两处近地面的固定立柱脚视差较好，也没有白斑的共同向下偏移。另一方面，白斑比立柱更靠近镜头，深度敏感度不同；这些静态点**不能排除**一种深度相关的 L154 位姿误差，也不能单凭 y 坐标差异判断鱼眼畸变。

对白斑另做了一个局部图像检查：分别从 L153 和 L155 裁片 LoMa 匹配估计到 L154 的局部单应，排除三条被审阅候选周围 20 像素内的点；匹配限制在粗白斑中心 120 像素内、得分 >0.6，再用 RANSAC 3 像素筛选。L153→154 为 **43/29** 条输入/内点，L155→154 为 **50/37**。两条白斑候选的单应转移位置离 L154 LoMa 点，前一方向约 **1.7–2.0 像素**，后一方向约 **0.1–0.3 像素**；原图 96/160 像素局部灰度与梯度相关性搜索给出的最佳平移分别为 **0 至 −2 像素**、**0 像素**。[逐点单应与相关性结果](../work/basketball/global_sfm/sfm_l154_static_control_probe/comparison.json)。这支持白斑**局部图像对应**没有明显 5 像素整体错位；单应使用邻近 LoMa 点且鱼眼/视角会带来近似误差，仍不能证明两个点的精确物理身份。

目前能确认的是：在同一支撑帧下，白斑小区域与保存的 SfM 候选约有共同 **+4–5 像素 y 残差**，而不同深度固定结构没有同方向的统一残差；现有原生前端在这片地面还缺少**分布更广、物理点位明确**的多帧轨迹。原因仍可能是局部地面点位/成轨偏差、近距离位姿灵敏度、鱼眼内参误差或它们的组合。下一步应在球场另一侧和不同图像半径寻找至少两处独特固定地面标记，做同样的五帧裁片、原图审阅和 L154 留出。只有独立地面位置也支持一致修正，再隔离比较 L154 局部位姿与鱼眼内参候选，并用保留的静态点约束；当前白斑候选不升为高权重 BA 点。诊断状态见[静态对照标记](../work/basketball/global_sfm/sfm_l154_static_control_probe/DIAGNOSTIC_ONLY.json)和[裁片标记](../work/basketball/global_sfm/sfm_l150_crop_loma_probe/DIAGNOSTIC_ONLY.json)。

后续已核对球场右侧白线角点：它的 L154 有符号残差方向与白斑相反，但旧模型已经用同一物理角点建立 23 视图轨迹，不能算新增独立地面控制；简单单点位姿修正还会伤害静态或旧地面约束。详见[右侧角点与位姿假设核查](basketball_sfm_l154_right_corner_probe.md)。

复算入口（裁片 LoMa 首次执行需要 GPU；已有缓存时直接复用）：

```bash
PYTHONPATH=src .venv/bin/python -m geosunpre sfm-l150-crop-loma-probe --work work/basketball
PYTHONPATH=src .venv/bin/python -m geosunpre sfm-l150-crop-audit --work work/basketball --stage all
PYTHONPATH=src .venv/bin/python -m geosunpre sfm-l154-static-control-probe --work work/basketball --stage all
```
