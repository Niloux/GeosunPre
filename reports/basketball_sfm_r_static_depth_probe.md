# R277–281 近远静态结构的留出检查

[白线角点复核](basketball_sfm_court_white_line_probe.md)提示 R277 右角定位不稳，因此先检查同一批 R 异步曝光里与球场白漆无关的静态点。候选从已有 LoMa 缓存读取，要求三张原生鱼眼图上的三条两两匹配**严格闭环**，并排除已进入保存模型 3D 点的特征。先看[原图候选](../work/basketball/global_sfm/sfm_r_static_depth_probe/review/selected_native_static_cycles.jpg)和[R277 专项原图](../work/basketball/global_sfm/sfm_r277_static_probe/review/selected_native_static_cycles.jpg)，再冻结点位、标签和输入哈希；直到这一步后才读取保存视觉 SfM 的位姿及候选鱼眼内参计算留出。[R277–281 选择协议](../work/basketball/global_sfm/sfm_r_static_depth_probe/selection_protocol.json)、[已冻结的特征索引与坐标](../work/basketball/global_sfm/sfm_r_static_depth_probe/selection.json)、[R277 专项选择](../work/basketball/global_sfm/sfm_r277_static_probe/selection.json)。导航 pose 数值、LAS、M 和无对应 pose 记录的图像仍不参与。

R276–278 的 LoMa 匹配形成 588 条三边闭环，其中 161 条在选定上部图像区域且三帧均未建图；R277–279、R279–281 分别有 156、100 条。高分闭环中大量点落在树叶和灌木，不能仅靠匹配分数认定为精确的静态点。人工原图核对后，主证据只保留建筑边缘、固定篮架接头/底座、灯杆底座/灯头；树干底部靠近绿篱和重复窗格只作上下文。[R277 专项逐点留出](../work/basketball/global_sfm/sfm_r277_static_probe/audit.json)、[R278/R280 逐点留出](../work/basketball/global_sfm/sfm_r_static_depth_probe/audit.json)。

| 留出曝光 | 已冻结的主证据 | 原图留出误差 | 残差向量（预测减观测） |
| --- | --- | ---: | ---: |
| R277 | 篮架红色支架接头 | 2.26 px | (−0.49, −2.21) px |
| R277 | 灯杆底座 | 0.82 px | (−0.45, +0.69) px |
| R277 | 建筑立面边缘 | 1.80 px | (−1.79, +0.19) px |
| R277 | 灯头右上角 | 5.64 px | (−5.05, +2.52) px |
| R278 | 两个建筑边缘 | 0.79、1.66 px | 详见逐点结果 |
| R280 | 篮架底座、支架接头 | 1.23、3.29 px | (+1.23, −0.05)、(−0.46, −3.25) px |
| R280 | 高楼边缘、屋顶构件 | 1.49、2.32 px | (+1.35, +0.63)、(+2.31, +0.23) px |

每条三帧轨迹只用**两端**观测和保存的视觉 pose 三角化，再预测中间曝光。以上点均未作为保存 3D 模型的训练点，但相机位姿仍由同一模型的其它点优化过；因此结果是局部一致性诊断，不是独立的相机真值。建筑/篮架“远近”按原图场景区分，不把模型求出的深度当作已验证的尺度。R277 建筑立面两端视线夹角只有约 0.63°，其低误差对深度的约束很弱；篮架和灯杆结构的夹角约 10–12°，更有判别力。灯头 5.64 px 的单点偏差需要保留，但与篮架及灯杆底座的方向不同，不能组成统一的 R277 位姿修正。

R280 附近的两个篮架结构点与右侧场地角点相距约 225–240 原图像素，保存视觉模型对它们的中间帧预测分别只偏 1.23 和 3.29 px；建筑两个点也没有同向的大偏移。相较之下，[旧 3D 点到右角白线候选的投影](../work/basketball/global_sfm/sfm_court_white_line_probe/coverage.json)在 R277、R280 分别相差约 9.39、7.2 px，且右角白漆宽度与浅交角使物理点位本身含糊。两组数值不能直接相减当作 pose 误差，但**没有形成跨深度、同方向的大残差**；据此不实施局部 R pose 或鱼眼内参 BA，也不将这些角点设为高权重。全场 3DGS 训练视角的一致性问题仍未解决，不能凭本段的较小留出误差宣称重建质量改善。

复算：

```bash
.venv/bin/python -m geosunpre sfm-r-static-depth-probe --work work/basketball --stage all
.venv/bin/python -m geosunpre sfm-r277-static-probe --work work/basketball --stage all
```
