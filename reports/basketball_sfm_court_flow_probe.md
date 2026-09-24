# 篮球场 SfM：近场角点与三帧光流诊断

3DGS 的训练视角仍存在图像不一致；这一轮只检验在球场铺装上补充独立角点，能否提供值得加入 SfM 的近场同名点。试验使用 L120–122 和 L150–152 六张带 pose 记录的原生鱼眼图像，沿用上一轮隔离的拟合 179° 球场候选掩码。导航 pose、LAS、M 图像和无 pose 静态照片均不参与；保存的等权视觉 SfM 位姿仅用于**后验诊断**，不是候选点的真值或选择依据。全场相机解、LoMa 缓存、COLMAP/LFS 输入未改。

半分辨率图像上用 Shi–Tomasi 提角点，分别做首→中、首→末、中→末光流和反向跟踪。每条边往返误差须小于原图 1 px，两种到达末帧的路径须在原图 1 px 内闭合，且三帧都在球场颜色候选区域中。保留下来的点映回原生 `OPENCV_FISHEYE` 像素；保存的视觉位姿从首末帧三角化并查中帧误差。另在不改变选择结果的情况下，用 82 原图像素宽的灰度块，在跟踪位置附近比较更大图像块的相关峰。相关峰只能提示局部纹理歧义；视角变化也会降低相关性，不能把它当作对应点真值。完整阈值和输入哈希见[原始角点协议](../work/basketball/global_sfm/sfm_court_flow_probe_v2/protocol.json)与[平滑角点协议](../work/basketball/global_sfm/sfm_court_structural_probe/protocol.json)。

| 三帧 L 图像 | 检测角点：原始/平滑 | 图像三帧闭环：原始/平滑 | 旧视觉解三视图合格：原始/平滑 | 其中离原 LoMa 首帧特征 >5 px：原始/平滑 |
|---|---:|---:|---:|---:|
|120–122|3292 / 752|30 / 11|21 / 8|16 / 4|
|150–152|5000* / 880|51 / 27|33 / 20|26 / 14|

\* 原始角点达到设置的 5000 上限，并非图中只有 5000 个候选。平滑试验对检测和光流输入使用半分辨率 σ=2.5 px 的高斯平滑，以抑制细密格纹；其余选择阈值相同。它使角点和闭环数明显下降，仍未产生旧解几何合格的近场候选。[原始完整结果](../work/basketball/global_sfm/sfm_court_flow_probe_v2/results.json)与[平滑完整结果](../work/basketball/global_sfm/sfm_court_structural_probe/results.json)包含每条边的流跟踪、几何拒绝原因与图像块统计。

表中的“离原 LoMa 特征 >5 px”只说明首帧位置不在已有特征附近，不等于建立了新的正确物理点轨迹。

最相关的失败出现在 L150–152：原始方案有 18 个闭环点位于首帧 y≥0.6H，但没有一个通过旧视觉解的三视图检查；17 个在其三角化下深度非正。图像下方只是近场**位置代理**，其中也可能有肩膀等非地面像素。红色铺装的[八组三帧原像素裁剪](../work/basketball/global_sfm/sfm_court_flow_probe_v2/review/L00150_near_crops.jpg)能看到重复格纹与部分十字偏离相同线交点的风险。对上述 18 点，首→中跟踪位置的大块图像相关系数中位数为 0.138，而旧几何合格点为 0.727；首→末为 0.094 对 0.640。所查范围内的最佳相关峰距光流位置中位数分别为 52 和 37 原图像素。搜索范围有限，峰值位移不是点位修正量。旧视觉位姿也可能有误，所以这些证据只说明**不能直接接纳近场格纹闭环点**，不能单独判定所有点是错配，亦不能证明原 SfM 正确。

原始试验中，首→中、首→末、中→末各自满足 1 px 双向误差的角点数，L120 分别是 431、184、491；L150 是 834、341、862。跨两帧的首→末边尤其难保持精确。经过三边同时通过、三帧闭环和区域检查后分别只剩 30、51 点。旧视觉解合格的点在[首帧预览 L120](../work/basketball/global_sfm/sfm_court_flow_probe_v2/review/L00120.jpg)、[L150](../work/basketball/global_sfm/sfm_court_flow_probe_v2/review/L00150.jpg)中主要位于远处球场边界及背景，而非用户关心的近地面铺装。

两版已分别[标记暂不晋升：原始](../work/basketball/global_sfm/sfm_court_flow_probe_v2/NOT_PROMOTED.json)、[平滑](../work/basketball/global_sfm/sfm_court_structural_probe/NOT_PROMOTED.json)。目前没有证据表明仅靠密集角点和小窗口三帧光流能改善近场 SfM，更不能推断 LFS 训练视角会改善。下一步应先在球场标线交点、铺装色块边界和稀有局部缺陷上建立可辨识对应，并用大尺度图像块、局部形变与多视图几何交叉审核；得到足量可靠近场轨迹后，才值得做局部 SfM/BA 和冻结图像查询上的一致性比较。

复算入口：

```bash
uv run --no-sync geosunpre sfm-court-flow-probe --work work/basketball
uv run --no-sync geosunpre sfm-court-structural-probe --work work/basketball
```
