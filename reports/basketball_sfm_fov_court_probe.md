# 篮球场 SfM：近场铺装的有效域与前端试验

上一轮虚拟视图前端新增了许多建筑点，却很少增加球场铺装上的三帧可靠轨迹。本轮检查特征进入 LoMa 前的有效域，并在六张 L 图像上隔离测试放宽鱼眼前半球边界。**结论是当前 160° FOV 与底部整带设备掩码都会丢掉可见铺装，但只放回这部分图像并没有明显增加可靠球场轨迹；候选暂不替代全场 SfM。**

## 两层截断分别是什么

[现有掩码实现](../src/geosunpre/static_masks.py)在每张原图的 y≥0.8H 整带设为设备无效；[原生鱼眼有效域](../src/geosunpre/camera.py)另外限制为候选 160° 全视角。两者都只是当前前端的假设，并非设备或镜头标定真值。在保存的等权**视觉拟合**内参下，L 的 160°、175°、179° 径向边界约为 1253、1342、1365 原图像素。六张 L 图在 0.8H–1.0H 区间中，160° FOV 内每张只有 24,752 像素，而 179° FOV 内有 143,148 像素。因此不能把底部缺点全部归咎于整带设备掩码：160° 本身先排除了大部分底部图像。[FOV 轮廓预览 L120](../work/basketball/global_sfm/sfm_bottom_mask_probe/fov_review/L00120.jpg)和[L150](../work/basketball/global_sfm/sfm_bottom_mask_probe/fov_review/L00150.jpg)可直观看到边界与球场位置。

只按图像颜色与位置生成一个**供审查的球场候选区域**，在 L150 的拟合 160° 域内约有 253 万像素，而 160°–179° 环带另有约 43 万像素；L120 的对应新增约 6.9 万像素。[六图 FOV 审计](../work/basketball/global_sfm/sfm_bottom_mask_probe/fov_audit.json)保存各图数量。候选区域使用青蓝、红色铺装的 HSV 范围及下半画面位置，预览见[区域提议 L120](../work/basketball/global_sfm/sfm_hybrid_court_probe/court_review/L00120.jpg)与[L150](../work/basketball/global_sfm/sfm_hybrid_court_probe/court_review/L00150.jpg)；它会包含少量其他彩色物体，也会漏掉低饱和度的标线，不能当作地面真值。

不能继续机械地把 FOV 扩到镜头可见圆周。在当前 `OPENCV_FISHEYE` 逆投影实现中，拟合 L 相机从 160°–179° 的点射线回投数值正常；但从主点向下约 1400 px 时，已超出前半球可逆范围，逆投影再回投可偏离约 2736 原图像素。这个计算说明**当前实现不支持把所有可见圆内像素直接作为同一原生鱼眼相机的可信观测**；不说明真实镜头的物理 FOV 就是 179°。

## 隔离的 179° 球场区域对照

试验只用 L120–122、L150–152 六张原图：保留原 160° 有效域，只在 160°–179° 环带放行上述球场颜色候选；对底部整带的 `device` 标签，也仅恢复此候选中 y∈[0.8H,0.9H) 的像素。天空环带、镜头边框和大部分设备仍被排除。[候选掩码协议与逐图面积](../work/basketball/global_sfm/sfm_fov179_court_candidate/mask_audit.json)、[L150 原图叠加预览](../work/basketball/global_sfm/sfm_fov179_court_candidate/review/L00150.jpg)可以复核哪些像素真正新开放；旧掩码、全场特征和 LFS 训练输入未改。

用原生 JPEG、原生鱼眼坐标和原 LoMa 1536/3072 配置重提六张图，另存特征与匹配。所有保留的新点都在拟合 179° 范围内，并通过鱼眼射线回投检查，最大数值误差约 1e−12 px；有 3 个落在掩码像素边缘之外的点被排除。回投只证实计算自洽，不证实内参正确。[前端协议](../work/basketball/global_sfm/sfm_fov179_court_frontend_probe_v2/protocol.json)与[完整逐图/逐边/三帧结果](../work/basketball/global_sfm/sfm_fov179_court_frontend_probe_v2/results.json)保存同一保存视觉 pose 下的对照。

| L 三帧 | 每图旧 FOV 外候选点 | 三帧闭环轨迹：旧→候选 | 三视图几何合格：旧→候选 | 其中球场颜色候选区域：旧→候选 |
|---|---:|---:|---:|---:|
|120–122|49–82|237→249|189→198|52→55|
|150–152|273–340|308→323|257→278|37→42|

新开放区域确实产生了点，L150 单张球场候选特征由 1390 增至 1467，但能进入该区域三视图几何的增量仍很小。三帧检查要求三条高分匹配边的索引闭合，并用当前保存的视觉 pose 从首尾帧三角化，检查中间帧原图重投影；这些 pose 原本由旧前端拟合，因此并非候选 SfM 或独立真值。旧人工标签也只是粗标，未参与阈值选择或验收。

这一版已[标记暂不晋升](../work/basketball/global_sfm/sfm_fov179_court_frontend_probe_v2/NOT_PROMOTED.json)。下一步需要解决的是**球场铺装、标线和边界上的同名点精度及多视图错配**，并在可信的相机可逆区域内先建立明显更多的可靠轨迹；届时再做完整局部 SfM 与冻结查询的相同图像对照。单靠扩大 FOV 或增加检测点数，当前没有证据能修复 LFS 训练视角的一致性。

复算入口：

```bash
uv run --no-sync geosunpre sfm-court-region-probe --work work/basketball
uv run --no-sync geosunpre sfm-bottom-mask-probe --work work/basketball
uv run --no-sync geosunpre sfm-fov-audit --work work/basketball
uv run --no-sync geosunpre sfm-fov179-masks --work work/basketball
uv run --no-sync geosunpre sfm-fov179-frontend-probe --work work/basketball
```
