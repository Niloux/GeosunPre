# 篮球场 SfM：定向 LoMa 与球场平面运动核查

上一轮原生鱼眼半分辨率角点/短窗口 LK 很难在重复铺装上给出可靠的近场同名点。本轮先检验球场结构点，再把 LoMa 检测集中到朝向场地的单个临时针孔视图；最后以**整块球场的图像平面运动**核对保存的视觉 SfM 位姿。**结论：定向 LoMa 明显增加球场区域三帧轨迹，但可目视确认的新增近场点仍不足以直接加权；球场平面运动跨三帧自洽，当前视觉位姿在这两个短片段也大体相容。此次不改全场 SfM 或 LFS 输入。**

只用 L120–122、L150–152 六张带 pose 记录的原生鱼眼图像。L/R/M 都按独立曝光处理；本试验仅取 L，绝不以同帧编号推定 L/R 同时曝光。导航 pose、LAS、M、无 pose 静态照片和旧粗人工标签不参与拟合或验收。现有拟合 `OPENCV_FISHEYE` 内参只负责把临时视图映回原图；保存的等权视觉 SfM 位姿只用于候选生成后的诊断，二者均非真值。

## 从结构点到定向 LoMa

先在粗尺度 Lab 色差边界附近提 SIFT。只用色块边缘时，L120/L150 的高分三帧描述子闭环仅为 1/0；加入低饱和度白线候选并降低特征尺度门槛后成为 2/8，旧视觉解三视图检查合格 1/4。[第一版结构区域](../work/basketball/global_sfm/sfm_court_structure_probe/review/L00150.jpg)漏掉白色弧线，[第二版区域](../work/basketball/global_sfm/sfm_court_structure_probe_v2/review/L00150.jpg)虽覆盖弧线，却也把大量格纹和少量背景当成白线候选。第二版 L150 只有 1 条图像下方合格轨迹，且距原 LoMa 特征不超过 5 原图像素；它不能提供新的近场约束。[第一版结果](../work/basketball/global_sfm/sfm_court_structure_probe/results.json)、[第二版结果](../work/basketball/global_sfm/sfm_court_structure_probe_v2/results.json)均保留，不将这两套结构点并入 SfM。

接着从每张 L 原图按当前鱼眼候选内参生成 **1280×1280、80°、朝下 45°** 的临时针孔视图，只在该视图采 LoMa 特征并匹配。视图上的所有候选映回原生 `OPENCV_FISHEYE` 像素，仍沿用每次曝光各自的相机位姿；虚拟视图绝不是新的 SfM 相机。[L120 场地视图与有效区域](../work/basketball/global_sfm/sfm_court_down_loma_probe/review/L00120_view.jpg)、[L150 对应预览](../work/basketball/global_sfm/sfm_court_down_loma_probe/review/L00150_view.jpg)可检查视图覆盖。区域候选仍来自上一轮粗 HSV 掩码，会包含局部草地，不能称为精确地面分割。[协议](../work/basketball/global_sfm/sfm_court_down_loma_probe/protocol.json)冻结六张原图、掩码、旧视觉模型、视图和 LoMa 参数。

| L 三帧 | LoMa ≥0.9 三边索引闭环 | 旧视觉解三视图合格 | 合格且首帧距原 LoMa 特征 >5 px | 其中首帧 y≥0.6H |
|---|---:|---:|---:|---:|
|120–122|162|148|108|5|
|150–152|74|48|42|1|

这些是[新视图结果](../work/basketball/global_sfm/sfm_court_down_loma_probe/results.json)中的数量，不能把“>5 px”解释为不同的真实物理点。原图预览[L120](../work/basketball/global_sfm/sfm_court_down_loma_probe/review/L00120_tracks.jpg)、[L150](../work/basketball/global_sfm/sfm_court_down_loma_probe/review/L00150_tracks.jpg)显示不少轨迹确实落在球场铺装上；但对全部 5/1 条靠下、合格且位置新颖的轨迹做[三帧原像素裁剪 L120](../work/basketball/global_sfm/sfm_court_down_loma_probe/review/L00120_near_novel_crops.jpg)、[L150](../work/basketball/global_sfm/sfm_court_down_loma_probe/review/L00150_near_novel_crops.jpg)后，点多在蓝色铺装边缘或红色重复格纹上，不能凭这些裁剪确认逐像素同名点。这六条不应作为高权重近地面 BA 约束。

另做的[短窗口 LK 审计](../work/basketball/global_sfm/sfm_court_down_loma_probe/lk_audit.json)在所有轨迹上三边同时通过数为零，但**这项否定结果不具判别力**：匹配位移中位数在 L120 的 AB/AC/BC 分别约 142/421/290 原图像素，在 L150 约 346/736/379 像素，已远超所用 21 px 窗口、三层金字塔的可靠捕获范围。不能据 LK 失败断言 LoMa 错配。[审计状态](../work/basketball/global_sfm/sfm_court_down_loma_probe/INCONCLUSIVE.json)单独记录这个限制。

## 不依赖单个格纹点的平面运动

对每组 L 图的 AB、AC、BC 高分 LoMa 匹配，在**虚拟针孔图像坐标**各自用 MAGSAC 拟合平面单应变换；不读入视觉 pose。再用独立的粗 HSV 红/绿/蓝色块边界检查整幅球场的扭正效果。[平面协议](../work/basketball/global_sfm/sfm_court_plane_probe/protocol.json)、[逐边结果](../work/basketball/global_sfm/sfm_court_plane_probe/results.json)和[叠加预览 L150 的 AC](../work/basketball/global_sfm/sfm_court_plane_probe/review/L00150_ac.jpg)、[L120 的 AC](../work/basketball/global_sfm/sfm_court_plane_probe/review/L00120_ac.jpg)可复核。

| L 三帧 | AC 单应内点 / 高分匹配 | AC 内点误差中位数 | 三个单应组成闭环的网格差中位数 | AC 主要色块 IoU：变换后（未变换） |
|---|---:|---:|---:|---:|
|120–122|207 / 255|1.59 px|1.08 px|绿 0.990（0.552）；蓝 0.777（0.476）|
|150–152|109 / 160|1.61 px|0.78 px|红 0.998（0.539）；绿 0.983（0.172）|

这里的像素单位均为 **1280×1280 临时视图像素**，不能直接读作原生鱼眼 1 px。颜色 IoU 由粗分割得到，面积大的色块会使它偏高；叠加图和边界统计比单个 IoU 更适合查看局部偏差。单应闭环和色块对齐表明此处存在可利用的整体平面约束，但重复格纹可能支持一致的错误平移，尚不能视为相机真值。

最后，用同一现有视觉 SfM 位姿和当前拟合内参，寻找**同一球场平面**在三对图像中所能产生的最佳单应变换，并与上述图像单应比较；只检查兼容性，不优化相机。[三边共用平面结果](../work/basketball/global_sfm/sfm_court_plane_probe/common_pose_audit.json)的网格位移差中位数：L120 的 AB/AC/BC 为 1.32/1.61/1.13 视图像素；L150 为 1.19/2.21/2.91，后者 BC 的 P90 为 8.89。旧位姿在两个短片段内**并未与球场整体运动明显冲突**，但存在局部尾部差异，且姿态、内参、平面假设的误差不能仅凭此拆分。逐边各自拟合平面的[对照](../work/basketball/global_sfm/sfm_court_plane_probe/pose_audit.json)也保存。

因此，[定向 LoMa 点轨迹暂不晋升](../work/basketball/global_sfm/sfm_court_down_loma_probe/NOT_PROMOTED.json)，[平面结果仅用于诊断](../work/basketball/global_sfm/sfm_court_plane_probe/DIAGNOSTIC_ONLY.json)。此处不做只针对两组三帧的局部 BA，也不宣称 LFS 训练视角已改善。下一步应沿 L 轨迹及有实际视野重叠的 R 图像扩展同样的图像平面/边界检查，先找出明显不兼容的区段，再对那些区段构建隔离的 SfM 位姿候选并用相同训练图像的对齐误差评估。L/R 虽在同一设备上，仍按异步曝光逐张求位姿。

复算入口：

```bash
uv run --no-sync geosunpre sfm-court-structure-sift-probe --work work/basketball
uv run --no-sync geosunpre sfm-court-down-loma-probe --work work/basketball --stage prepare
uv run --no-sync geosunpre sfm-court-down-loma-probe --work work/basketball --stage match
uv run --no-sync geosunpre sfm-court-down-loma-review --work work/basketball
uv run --no-sync geosunpre sfm-court-down-loma-audit --work work/basketball
uv run --no-sync geosunpre sfm-court-plane-probe --work work/basketball
uv run --no-sync geosunpre sfm-court-plane-pose-audit --work work/basketball
uv run --no-sync geosunpre sfm-court-plane-common-pose-audit --work work/basketball
```
