# L154 右侧白线角点与白斑位姿假设核查

在[白斑与静态结构同支撑核查](basketball_sfm_l154_patch_control.md)之后，继续寻找另一处图像位置不同、物理身份明确的近地面标记。L151–155 右侧绿场与蓝色缓冲区的白线直角交点连续可见；本轮直接在原生 3000×4000 鱼眼原图中拟合两条白漆中心线的交点，先冻结像素，再读取保存的等权视觉 SfM 模型。输入导航 pose 数值、LAS、M 和无 pose 记录图像不参与；没有修改 BA、COLMAP 导出或 LFS 输入。

原图白线连通域、两条长臂与冻结交点见[五帧复核图](../work/basketball/global_sfm/sfm_l154_right_corner_probe/review/frozen_corner_and_mask.jpg)。检测只用各帧原图和预选的粗窗口；[图像选择协议](../work/basketball/global_sfm/sfm_l154_right_corner_probe/selection_protocol.json)与[逐帧阈值、种子扰动结果](../work/basketball/global_sfm/sfm_l154_right_corner_probe/selection.json)在 pose 审计之前冻结。

| 帧 | 白线中心交点原图像素 | 已通过图像门限的白色阈值最大位移 | 窗口中心 ±10 px 扰动最大位移 |
|---|---:|---:|---:|
| L151 | (2593.37, 2015.05) | 13.10 px | 3.31 px |
| L152 | (2601.74, 2064.60) | 仅基准阈值通过 | 0.63 px |
| L153 | (2648.56, 2088.64) | 3.88 px | 0.21 px |
| L154 | (2735.76, 2148.66) | 0.43 px | 0.27 px |
| L155 | (2784.04, 2162.05) | 0.20 px | 0.31 px |

L151 的宽松阈值会把交点移动 13.1 像素，不能当成精确五视图控制；L152 缺少第二个通过图像门限的阈值。以下严格只用 L153/L155 支撑、L154 留出。以保存的候选鱼眼内参和视觉位姿三角化，支持射线夹角 **7.61°**，L154 的“预测 − 图像检测”残差为 **(−0.34, −2.43) px**、模长 **2.45 px**。[逐帧留出与旧模型邻近点](../work/basketball/global_sfm/sfm_l154_right_corner_probe/audit.json)。12 组通过图像门限的 L153/L154/L155 阈值组合，L154 残差 y 始终在 **−3.16 至 −2.00 px**，x 在 **−2.57 至 +0.11 px**；见[完整敏感性结果](../work/basketball/global_sfm/sfm_l154_right_corner_probe/sensitivity.json)。与同支撑下白斑候选 A/B 的 **+4.67/+4.80 px y** 偏移方向不同，因此“给 L154 整张图施加相同二维平移”不能同时解释两处。

但右侧角点**不是全新的独立 SfM 约束**。保存模型的点 `25226` 在 L151–154 距离几何交点 3.93–5.41 像素，原图审阅显示它跟踪的就是同一个白线角点；它已有 **23 视图 L/R 轨迹**，保存模型平均重投影误差约 **1.48 px**。[点 `25226` 的全轨裁片](../work/basketball/global_sfm/sfm_l154_right_corner_probe/review/mapped_point_25226_all_views.jpg)。旧 LoMa 特征相对白漆中心线交点在 L151–154 约向左 2.5–3.6 像素、向下 2.4–4.2 像素，说明厚白线上的“特征中心”和“漆线中心线交点”本来就有不同像素定义。保存的 L154 位姿也曾受这条旧轨迹优化；上面的 2.45 像素只是另一种图像定位约定下的局部兼容性检查，不能当独立真值。

为检查白斑偏移是否至少能由一种简单 L154 位姿扰动解释，固定 L153/L155 三角化点，只让 L154 发生一次**最小范数**旋转或平移，各自仅以白斑候选 A 的两个像素残差求解，然后检查候选 B 和其它点。以下都是[反事实投影](../work/basketball/global_sfm/sfm_l154_right_corner_probe/pose_hypotheses.json)，没有运行 BA，也没有写回位姿。

| L154 点位误差 | 原模型 | 仅旋转拟合白斑 A | 仅平移拟合白斑 A |
|---|---:|---:|---:|
| 白斑 A（拟合点） | 5.02 px | 0.00 px | 0.00 px |
| 白斑 B（相邻、相关） | 5.08 px | 0.17 px | 0.36 px |
| 黑色立柱脚 | 0.38 px | 4.87 px | 0.94 px |
| 灰色立柱脚 | 0.90 px | 4.03 px | 1.27 px |
| 右侧白线中心交点 | 2.45 px | 3.04 px | 4.82 px |
| 旧 23 视图点 `25226` 的 L154 观测 | 2.81 px | 3.55 px | 5.16 px |

最后一行由[保存长轨迹影响核查](../work/basketball/global_sfm/sfm_l154_right_corner_probe/mapped_impact.json)给出。旋转假设显著伤害静态结构；平移假设能够利用深度差异让立柱变化较小，却明显伤害已有的球场角点轨迹。单个不精确白斑候选只有两个像素约束，最小范数选择本身任意，且白斑 A/B 高度相关；这些数字**不能排除**更复杂的多帧位姿、鱼眼内参或点位联合误差，也不能批准任何 pose 更新。

本轮找到的是一个真实、容易辨认的右侧地面角点，同时确认旧 SfM 已有它的长轨迹。白斑附近仍缺少同等可靠、空间独立的精确轨迹；继续给白斑点高权重或按单点修 L154 都会有明显过拟合风险。下一步更值得做的是先把白斑附近的物理点位延长并复核到更多曝光，或从其它时段已找到但旧模型未建图的白线角点补充真正新的近地面约束，再做隔离 BA 候选与静态/长轨迹留出检查。当前仍为[诊断状态](../work/basketball/global_sfm/sfm_l154_right_corner_probe/DIAGNOSTIC_ONLY.json)。

复算入口：

```bash
PYTHONPATH=src .venv/bin/python -m geosunpre sfm-l154-right-corner-probe --work work/basketball --stage all
```
