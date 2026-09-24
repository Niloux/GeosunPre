# 篮球场 SfM：标注语义修正后的前端试验

用户补充说明了[旧标注的实际语义](../work/basketball/global_sfm/static_validation/annotation_semantics.json)：`dynamic` 主要是树叶/植被上难以确认的点，不代表已确认运动；`wrong_match` 是目视发现的明显像素错位，并未标出全部不准的点；大多数 `same_static` 像素位置没有精调。因此旧人工点不能充当精确同名点真值，也不能据“341/342 个坐标未改动”推断 LoMa 已有足够精度。既有数值文件均保留，但解释已在[旧核查报告](basketball_static_validation.md)和[前端证据报告](basketball_sfm_frontend_evidence.md)修正。

当前前端在原图缩至长边 1536 后运行 LoMa 检测，把坐标映回 3000×4000 原图；匹配再经过鱼眼相机感知的两视图 E 验证与 COLMAP 成轨。这些步骤可以产生可用视觉模型，**但几何内点与高分匹配仍不是高精度像素对应的保证**。本轮在不使用导航 pose、LAS、旧人工标签或 LFS 的条件下，补了三个可复现的探索性试验。

## 原图双向跟踪与 LoMa 对应

[10 对相邻同镜头图像](../work/basketball/global_sfm/sfm_lk_probe/protocol.json)中，先取 LoMa 分数 ≥0.9 且已进入当前训练数据库两视图几何的匹配，再在原生 JPEG 缩半的图像上做两个方向的 LK 跟踪，各方向的回环误差都需 <1 **原图像素**。仅对这样的局部可重复匹配比较 LK 位置与 LoMa 目标点；[逐对统计](../work/basketball/global_sfm/sfm_lk_probe/results.json)保留无法跟踪的分母。

|镜头|所选相邻图像对|高分训练几何匹配|双向可重复|其中 LK 与 LoMa 相差 >2 原图 px|
|---|---:|---:|---:|---:|
|L|5|2,009|738|25.3%|
|R|5|1,671|77|28.6%|

R 的 77 个可重复点中有 65 个来自第 65→66 帧，其余图像对 LK 大多失败；不能把 R 的比例推广到整段。光流受视差、遮挡、曝光与重复纹理影响，**双向稳定仍不等于物理真值**。但该检查确实反驳了“通过几何筛选的 LoMa 点位自然就够精准”的推断，也提示先看图像对的可跟踪性，再讨论如何改点位。

## 保守训练点位候选与局部 BA

在 70–94 帧 50 张 L/R 图像中，只给已映射的训练观测生成相邻**同镜头**图像的 LK 建议；双向回环稳定、LoMa 与 LK 相差 ≤4 px，且至少两个不同邻图给出的建议在 1 px 内一致时，仅应用共识位移的一半，最大实际改动 1 px。[冻结协议](../work/basketball/sfm_lk_refinement/normal_070_094/protocol.json)和[逐观测审计](../work/basketball/sfm_lk_refinement/normal_070_094/lk_audit.json)锁定源模型、原图、阈值和隔离。48 对相邻图像共有 26,982 个共同映射点候选，5,369 个双向稳定，最终修改 **1,038 / 55,754 条训练观测**，覆盖 45 张图。留出整条原始轨迹未进入 LK 建议或 BA；点轨迹拓扑、L/R 内参保持不变，每张曝光位姿与三维点重新优化。

两组 BA 都收敛。用相同的冻结原始 LoMa 留出查询评估，结果如下；这些查询也不是精确像素真值。

|指标（原图 px）|等权基线重拟合|LK 训练点位候选|
|---|---:|---:|
|高置信闭合三图 832 查询 median / P90|1.6437 / 5.4549|1.6429 / 5.4517|
|其中 ≤2 px|59.25%|59.13%|
|两组共同合格支持 774 查询 median / P90|1.54718 / 4.63297|1.54721 / 4.62693|
|共同支持配对误差中位变化|—|+0.00016|

[完整对照](../work/basketball/sfm_lk_refinement/normal_070_094/comparison.json)还保存 L/R 与原图半径分组。这样保守、仅覆盖约 1.9% 观测的小位移方案**没有可见几何收益**，已[标记暂不晋升](../work/basketball/sfm_lk_refinement/normal_070_094/NOT_PROMOTED.json)。它不能代表前端错配和弱纹理区域问题已经解决。

## LoMa 检测分辨率 1536 与 2048

GPU 空闲后，对 4 对相邻同镜头图像另提取长边 2048 的 LoMa 特征（3072 点/图，独立缓存），与原 1536 配置分别运行同一鱼眼 E 检验和 LK 检查。[协议](../work/basketball/global_sfm/sfm_resolution_probe/protocol.json)与[结果](../work/basketball/global_sfm/sfm_resolution_probe/results.json)记录输入哈希、图像对和新缓存 token。两种分辨率选出的点不同，指标**不是同名点配对误差**。

|图像对|E 验证后高分匹配 1536→2048|E 拟合内部留出误差 median / P90（px）1536→2048|双向 LK 可重复数 1536→2048|
|---|---:|---:|---:|
|R 65→66|399→431|1.835 / 7.138 → 1.929 / 10.590|70→74|
|R 70→71|305→317|1.292 / 8.092 → 1.267 / 9.307|10→8|
|L 120→121|580→564|1.202 / 7.067 → 1.187 / 6.400|274→275|
|L 150→151|788→689|0.708 / 3.475 → 0.522 / 3.442|549→469|

2048 在两对 L 图像的局部指标有改善，但 R 65→66 的 E 误差尾部变差，R 70→71 的 LK 可重复点极少；L 150→151 的几何高分匹配数也下降。四对图像不足以支持全场切换分辨率。本轮未替换任何生产特征缓存、全场 SfM 模型或 LFS 输入。

## 虚拟针孔视图提特征，再映回原图

进一步在相同四对相邻图像上，按当前**视觉估计**的 L/R 鱼眼内参把每张原图投成正向、水平 ±45°、俯仰 ±45° 五个 100°、1024² 像素的视图；每视图运行 LoMa 检测与描述，再把点映回鱼眼原图、应用原有 FOV/语义掩码、3 px 去重，最终每张保留 3072 点。虚拟视图仅是前端采样方式，**不增加 SfM 相机，也不改变 L/R 异步曝光模型**。内参决定临时采样和映回，但并非真值。[冻结协议](../work/basketball/global_sfm/sfm_virtual_view_probe/protocol.json)和[原始逐对结果](../work/basketball/global_sfm/sfm_virtual_view_probe/results.json)保留输入哈希、点位与匹配缓存。

对比原生 LoMa，四对图像的几何验证后高分匹配数均增加，例如 L120→121 为 580→818，L150→151 为 788→1270。但**各自重新拟合**的 E 在原始匹配内部留出的 P90 误差四对均升高：R65 7.14→10.83、R70 8.09→10.90、L120 7.07→8.35、L150 3.47→5.01 原图像素。这些是不同位置的对应，不是逐点误差对照。

[固定视觉模型核查](../work/basketball/global_sfm/sfm_virtual_view_probe/fixed_model_audit.json)用同一组已保存的视觉位姿评估两套前端，避免把每套匹配各自重拟合的 E 当成直接精度比较。L120 的下半原图高分匹配在固定 pose 下，E 误差中位数 0.93→1.13 px、≤3 px 比例 91.7%→89.1%。每张 L 图下半画面**特征**占比也下降：L120 45.3%→38.6%，L150 44.5%→30.7%。这些 pose 来自原始前端，会偏向它；下半画面只是一种屏幕位置代理，不等于经验证的地面分割。

为了检查“新增匹配能否成轨”，又对 R65–67、R70–72、L120–122、L150–152 四组三连续同镜头图像做[三帧试验](../work/basketball/global_sfm/sfm_virtual_cycle_probe/protocol.json)。要求 LoMa ≥0.9 的 AB、AC、BC 三条边在三张图像各自的特征索引上形成一致闭环；再用保存的视觉 pose 从第一、三帧三角化，要求支持点正深度、视差 ≥1°、最大重投影 ≤3 px，检查中间帧投影 ≤3 px。结果和所有分母见[三帧结果](../work/basketball/global_sfm/sfm_virtual_cycle_probe/results.json)。

|三帧|高分闭环轨迹：原生→虚拟|满足三视图几何：原生→虚拟|其中原图下半画面：原生→虚拟|支持合格轨迹的中间帧误差中位数 / P90（px）：原生→虚拟|
|---|---:|---:|---:|---:|
|R65–67|130→204|96→169|35→74|1.49 / 3.50 → 1.51 / 3.74|
|R70–72|138→138|59→76|48→46|1.66 / 3.15 → 2.36 / 4.09|
|L120–122|237→395|189→336|60→61|1.05 / 2.51 → 0.82 / 1.73|
|L150–152|308→774|257→716|25→39|0.98 / 1.89 → 1.07 / 2.09|

原图上的[闭环轨迹分布预览 L120](../work/basketball/global_sfm/sfm_virtual_cycle_probe/review/L00120.jpg)、[L150](../work/basketball/global_sfm/sfm_virtual_cycle_probe/review/L00150.jpg)、[R65](../work/basketball/global_sfm/sfm_virtual_cycle_probe/review/R00065.jpg)显示：L 两组新增闭环主要在建筑，球场铺装区域增长有限；R65 的下半画面主要是庭院地面和植被，不能代替球场验证。预览中黄色为上半画面，青绿色为下半画面，白线是画面中线；画的是**三帧闭环、尚未通过三维几何筛选**的点。三帧检查也不能证明植被点静止或像素定位正确，且使用了原始前端训练出的固定 pose，并非候选重建对照。

因此这版全场虚拟视图前端已[标记暂不晋升](../work/basketball/global_sfm/sfm_virtual_cycle_probe/NOT_PROMOTED.json)：有扩大匹配和成轨覆盖的潜力，但缺少对球场地面与训练视角一致性的明确增益。下一步应针对球场铺装、标线和边界做**区域可核查的多视图点位选择与错配筛选**，在这一步确实提升下半画面静态结构的可靠轨迹之后，才值得进行局部 SfM 重建及冻结查询对照。最终像素精度还需另建认真逐点校正、覆盖球场标线、建筑、L/R 和鱼眼外圈的盲审集；旧粗标不用于为新方案调阈值后自我验收。

后续已针对球场可见区域检查 160° FOV 与底部设备整带掩码，并试跑六张 L 图像的隔离 179° 球场区域前端；三视图球场轨迹增益仍有限。见[近场铺装有效域与前端试验](basketball_sfm_fov_court_probe.md)。

复算入口：

```bash
PYTHONPATH=src .venv/bin/python -m geosunpre sfm-lk-probe --work work/basketball
PYTHONPATH=src .venv/bin/python -m geosunpre sfm-lk-refinement --work work/basketball --stage all
uv run --no-sync geosunpre sfm-resolution-probe --work work/basketball
uv run --no-sync geosunpre sfm-virtual-view-probe --work work/basketball
uv run --no-sync geosunpre sfm-virtual-view-audit --work work/basketball
uv run --no-sync geosunpre sfm-virtual-cycle-probe --work work/basketball
```
