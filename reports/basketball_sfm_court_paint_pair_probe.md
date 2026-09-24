# 全场原生鱼眼图中的场地双角轨迹

上一轮在 L231–L242 找到一条旧 LoMa 前端未覆盖的红色场地角点轨迹。这轮把**图像侧**的提案扩展到全部 536 张有 pose 记录的 L/R 图像，寻找同一块红色铺装区域所连的两个红/绿/蓝交点，并只在相机内连续曝光上延长轨迹。L/R 不按同编号同步配对；M、无 pose 的图像、输入 pose 数值和 LAS 均未进入提案。既有等权视觉 SfM 位姿与拟合候选鱼眼内参只在原图精定位完成后用于留出检查，不当作真值。[扫描协议](../work/basketball/global_sfm/sfm_court_paint_pair_probe/scan_protocol.json)、[全图逐帧扫描](../work/basketball/global_sfm/sfm_court_paint_pair_probe/scan.json)、[原图与视觉解检查](../work/basketball/global_sfm/sfm_court_paint_pair_probe/audit.json)、[诊断状态](../work/basketball/global_sfm/sfm_court_paint_pair_probe/DIAGNOSTIC_ONLY.json)。本轮未新增 BA 约束，也未修改 COLMAP/LFS 输入。

四分之一原图分辨率的 HSV 三色交集只给出粗搜索框；候选角点必须邻接同一块面积至少 2000 像素的红色连通区域，两角水平相隔 40–300 像素。同相机相邻曝光若任一角移动超过 150 像素则断开；仅审计至少 5 帧的段落。原生 3000×4000 图像再在各粗框内通过三色内部距离最小化定位角点。扫描得到 **L 106 张、R 46 张**有双角提案，形成 **14 段**长度至少 5 帧的连续轨迹；其中 **12 段**完成连续原图定位。两段未形成合格定位段，主要因为角点贴近图像边界；L230–243 中的 L232 也有一个角点落在精定位搜索边上，审计便从 L233 连续段重新起算。[L 段首/中/末原图接触表](../work/basketball/global_sfm/sfm_court_paint_pair_probe/review/L_segments_contact.jpg)、[R 接触表](../work/basketball/global_sfm/sfm_court_paint_pair_probe/review/R_segments_contact.jpg)标出左角红圈、右角蓝圈。

以下四条 **L 左角**是图像中可沿同一红色区域核对、且旧模型未覆盖的近地面候选：全部观测在保存的语义掩码和候选 160° 鱼眼有效域内有效，角点 **16 原图像素**内没有旧模型已建图 3D 特征。表中的留出误差是只用该段第约 1/4 和 3/4 位置的两张图三角化，以**现有视觉位姿及候选内参**预测其他帧；不是 BA 后残差，也不证明 pose 是真值。[逐帧误差与旧轨迹邻近度](../work/basketball/global_sfm/sfm_court_paint_pair_probe/audit.json)。最后一列是保持色相范围不变、将 HSV 饱和度下限设为 60/80/100、亮度下限设为 50/60/80 后，每帧检测位置最大移动量的中位数及全段最大值；它衡量**阈值敏感度**，并非真实定位误差。[逐帧九种阈值结果](../work/basketball/global_sfm/sfm_court_paint_pair_probe/stability.json)。

| L 连续段 | 左角留出帧数 | 留出中位 / 最大原图像素 | 两支持射线夹角 | 阈值位移中位 / 全段最大原图像素 | 原图裁剪 |
| --- | ---: | ---: | ---: | ---: | --- |
| 125–129 | 3 | 2.85 / 3.69 | 9.6° | 3.2 / 4.1 | [核对](../work/basketball/global_sfm/sfm_court_paint_pair_probe/review/L125_129_native_crops.jpg) |
| 172–178 | 5 | 2.54 / 4.43 | 13.1° | 4.1 / 11.0 | [核对](../work/basketball/global_sfm/sfm_court_paint_pair_probe/review/L172_178_native_crops.jpg) |
| 233–243 | 9 | 1.91 / 3.23 | 33.2° | 2.0 / 2.2 | [核对](../work/basketball/global_sfm/sfm_court_paint_pair_probe/review/L233_243_native_crops.jpg) |
| 252–259 | 6 | 1.75 / 4.89 | 22.3° | 5.4 / 9.5 | [核对](../work/basketball/global_sfm/sfm_court_paint_pair_probe/review/L252_259_native_crops.jpg) |

只有 **L233–243 左角**在这组阈值扰动下全段最大位移仍不超过 2.2 像素，是当前最稳的补充候选。L125 左角的角度基线偏小；L172、L252 有若干帧明显受阈值影响，不能因为旧 pose 留出误差小就直接赋高权重。L125–129 的**右角**也在 5 张图中均无 16 像素内的旧 3D 点，留出误差中位 2.41、最大 4.70 原图像素；但支持射线夹角仅 7.3°，逐帧阈值位移可达 14.9 像素，应视为视差和定位都较弱的补充候选。L137–144 左角虽同样未被旧模型覆盖，留出中位 5.19、最大 9.73 像素，需先复核像素定位。上述距离只检查**角点邻域**，不是整张图的特征密度或训练质量指标。角点少且集中于彩色场地区域，即便已分散到多个时间段，也不足以单独约束全场 536 张曝光。

自动选出的“同一红区双角”仍会接上篮架立柱而产生假点。例如 L48–53、L103–107 的一侧标记落在球架附近；其多帧留出中位分别约 31、59 原图像素。R214–225 的左侧角点在接近画面边缘、经过红色区域时也无法保持可信身份，旧视觉解下留出中位约 267 像素。[R214–225 原图裁剪](../work/basketball/global_sfm/sfm_court_paint_pair_probe/review/R214_225_native_crops.jpg)可检查这一失效。它们不能转成高权重约束。即使某条候选与视觉 pose 不一致，也不能只凭该残差将责任归于位姿；错认物理点、白线交点定位、鱼眼内参偏差都可能造成同类数值。

**R267–281 值得单独复核**：[两角原图裁剪](../work/basketball/global_sfm/sfm_court_paint_pair_probe/review/R267_281_native_crops.jpg)显示相同红色场地区域贯穿 15 次曝光，但现有视觉解下左角留出中位 / 最大为 **6.31 / 20.10 px**，右角为 **12.69 / 49.34 px**；其间预测深度也变化很大。左角在 60%、右角在约 87% 的帧中没有 16 像素内已建图点。不过阈值扰动使左角位置单帧最多移动 **23 px**，右角逐帧最大位移的中位数约 **12 px**、最高 **23 px**；现有定位方法不够稳定，不能从大残差推出 R pose 错误。下一步应先在这些 R 原图上用白线几何与邻近唯一纹理交叉核对，再冻结训练角点与不参与 BA 的验证角点；若多条空间分散、不同深度的静态轨迹同时支持位姿变化，才实施隔离的逐曝光 SfM 优化，并用相同 536 张训练图检查 LFS 图像一致性。

上述白线几何复核已完成，参见[白线交会与旧轨迹核查](basketball_sfm_court_white_line_probe.md)。它显著修正 R 左角的 HSV 坐标，却发现 R 右角在浅交角和宽白漆处仍缺少唯一稳定点位；R 左角部分帧已有旧 3D 轨迹。因此这份双角提案的原始误差不能直接用于判定 R 位姿或加权 BA。

复算入口：

```bash
uv run --no-sync geosunpre sfm-court-paint-pair-probe --work work/basketball --stage scan
uv run --no-sync geosunpre sfm-court-paint-pair-probe --work work/basketball --stage audit
uv run --no-sync geosunpre sfm-court-paint-pair-probe --work work/basketball --stage review
uv run --no-sync geosunpre sfm-court-paint-pair-probe --work work/basketball --stage stability
```
