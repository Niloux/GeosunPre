# COLMAP 与 PLY 交付

交付直接位于 `--output` 根目录，下游定制 LFS 训练的数据路径即为该目录。

```text
OUTPUT/
  images/{L,R}/**.JPG
  masks/{L,R}/**.JPG.png
  weights/{L,R}/**.JPG.png     # 默认启用静态掩码且光度校准通过时生成（320 长边单通道连续漫反射权重）
  depths/{L,R}/**.JPG.png       # 默认生成，--no-las-supervision 省略
  normals/{L,R}/**.JPG.png      # 默认生成，--no-las-supervision 省略
  sparse/0/{cameras,images,points3D,rigs,frames}.bin
  sparse/0/points3D.ply
  roi.json                    # 默认随 PLY 生成，--no-roi 或 --no-ply 时省略
  manifest.json
  quality.json
  initialization.json
  supervision.json            # 默认生成，--no-las-supervision 省略
  trainer.json                # 导出时按场景规模与本机显存自适应生成
  # masks 默认为持续人物及表面/室内天空误检改回静态后的掩码，--no-static-masks 时为 SAM3 掩码
  rolling_shutter.json        # 卷帘校正被采用时生成
  result.json                 # 各阶段报告，与工作目录 result.json 相同，不参与 check
```

COLMAP 小端二进制，`OPENCV_FISHEYE` 模型编号 5、8 个参数。相机 ID 1、2；每幅图像独立 frame，使用镜头对应的一传感器 rig。内部索引从 0 开始；仅导出已注册照片，按原输入顺序将图像/frame ID 连续重编号为 1 到注册数量，轨迹中的图像引用同步转换，内部缓存编号保持不变。

位姿 world-to-camera，COLMAP 四元数顺序 `w x y z`。图像名相对于 images，保留原图名、鱼眼畸变与全部特征编号；几何坐标只随可选有效域裁剪平移（不做去畸变或重采样），当静态掩码完成全局光度校准时，交付图像在物理线性空间执行 3 通道 KB8 径向暗角去除、3 通道白平衡对齐、安全界限内曝光拉齐与高光过曝通道软拐点压缩（`0.92` 起双曲滚降）并以 Q=97 RGB JPEG（4:4:4）写出，同时把残余相对曝光写入 EXIF（`--no-static-masks` 或 `--no-las-supervision` 时保持无损原图/DCT 裁剪）。图像名统一为 `镜头/文件名`（如 `L/xxx.JPG`）。卷帘快门校正被采用时，位姿为读出中线时刻的参考位姿，并附 `rolling_shutter.json`，见下节。点轨迹与图像引用相互一致，无三维点的特征用 uint64 最大值。下游须支持原生鱼眼模型。

points3D.bin 的 error 为最终所选模型在该点全部观测上的平均重投影距离，单位像素；联合优化后重新计算，交付回读时验证。

相机和稀疏点已在局部 LAS 坐标（局部 LAS 即输入坐标，ECEF LAS 为其转换后的导航 ENH 局部坐标，见[输入约定](input.md)），下游不再应用配准变换。`points3D.bin` 是清理并验证后的视觉稀疏点及轨迹（保持纯 LAS 锚定口径）；`points3D.ply` 主体保留已注册相机 50 m 视锥范围内的有效 LAS 点，按到最近相机的距离分档体素均值采样（10 m 内 1×、20 m 内 2×、40 m 内 4×、50 m 内 8× 基准体素；基准体素默认 0.05 m），并从清理前视觉模型补充距已注册图像 0.25–250 m、0.25 m 内无有效 LAS 覆盖的高置信静态多视图视觉点（不少于 3 视图、均值重投影不超过 1.5 px、视差至少 2°、无 0 掩码中心且至少 75% 且不少于 3 个视图的 5×5 邻域全为 255，按至少 0.10 m 网格去重并限制补充最多 10 万点），LAS 随同调整上限使 PLY 总点仍封顶 250 万；当启用 `roi.json` 且体素点超出该上限需要抽稀时，外围最多保留 5%（不超过 12.5 万点）作背景支撑，其余优先分配给 ROI 内部点，`initialization.json` 另记录 `las_points`、`visual_supplement_points`（启用 ROI 时另记录 `roi_las_points`、`outside_roi_las_points` 与 `roi_voxel_points_before_cap`），实际封顶记录于 `initialization.max_points`（默认随自适应 `max_cap` 联动为 `min(2500000, round(0.75 * max_cap))`，保证初始点云不顶死训练上限）。当静态掩码完成全局光度校准并生成 `cache/static_masks/diffuse_views.bin` 时，导出流式复用多视角校准结果在 25 m 方向分层共视邻域（至多 24 视角、LAS 深度遮挡剔除与高反光/阴影连续权重 `>= 64` 过滤）上按 `[25%, 65%]` 加权截断均值对 `points3D.ply` 逐点重着色为校准去反光漫反射 sRGB（`initialization.color = "calibrated_multi_view_diffuse_srgb"`，并在 `initialization.diffuse_recoloring` 记录重着色比率与中位有效视角数；不足 3 视角的点保留原色）。文件为二进制小端，每点 float32 XYZ、零法向量 XYZ、uint8 RGB，共 27 字节。PLY 与相机同一坐标，`initialization.coordinate_frame` 为 `input_las_xyz` 或 `ecef_to_navigation_enh`。LAS 头部范围（ECEF LAS 为转换后范围）的坐标绝对值须不超过 16777 m，使 float32 舍入不超过 1 mm，否则明确报错。下游需明确选择此 PLY 初始化 3DGS。

`roi.json` 默认随初始化 PLY 在 `export` 阶段自动生成（`--no-roi` 或 `--no-ply` 时省略）：基于 50 m 视锥内有效 LAS 在 1 m×1 m XY 格网的原始点占据（单格不少于 30 点），结合全部已注册相机轨迹连通光栅，经 2 m 圆形膨胀并从相机种子做 4 邻域连通筛选后提取外轮廓（多块时取外凸包），迭代 `approxPolyDP` 简化为 3–64 顶点的简单非自交 2D 多边形（兜底外切凸包或包围框），保证 100% 已注册相机严格落入内部；Z 范围取多边形内体素 LAS 分布与相机高度的稳健包络（`z_min = min(p001, cam_z_min - 2) - 1`，`z_max = min(max(p999, cam_z_max) + 15, max(p999 + 3, las_max + 1, cam_z_max + 3))`）。文件记录 `version: 1`、`type: "polygon_prism"`、`coordinate_frame`、`z_min`/`z_max`、`vertices_xy`（工作坐标二维顶点）、`bounds_min`/`bounds_max`、`params` 与 `stats`；定制 LFS 加载数据集时自动挂载为训练模型 `CropPolygon`，配合 [trainer.json](../configs/trainer.json) 在训练期降权外围并默认终态 `--crop` 裁切输出。`manifest.json` 的 `request.roi` 与顶层 `roi` 记录该文件身份，`check` 回读校验文件尺寸、多边形几何合法性与 100% 相机包含性。

掩码为与交付照片同尺寸的单通道 PNG：静态 255、天空 128、人物及有效域外 0。SAM3 grounding 在按主点居中的方形鱼眼 ROI 上决定人物与天空：person、robot、sky 主检测分数 ≥ 0.5 时有效，触及有效域外圈（10 格环带）的 person 与 robot 放宽至 ≥ 0.25，shadow 阈值 ≥ 0.25；人物检测有一半以上被分数不低于它的机器人检测覆盖时视为人形展品，保持静态；对人物执行半径 16 格椭圆闭运算桥接手臂穿戴配件（受 8% 有效域与 50% 人物面积约束并局部外扩 5 格）及填充人物围合的封闭小空隙（`<= 3%` 有效域且人物边界占比 `>= 35%`）；影子在 288 格掩码上须与保留人物相距不超过 24 格，且若单个影子检测面积超过整格网的 8% 或（超过 3% 且）超过接触人物连通块总面积的 3 倍，仅裁剪保留距人物 24 格内的局部区域，防止行人踩入建筑大阴影后连带抹掉整片静态地面（并在 `masks_profile.json` 记录 `shadows_clipped`）；远离人物的家具和展品阴影保持静态；人物与影子的并集经半径 2 格椭圆（5×5）外扩后放大到原图。天空为 sky 检测掩码的并集，不外扩，放大到原图后在有效域内置 128，人物与影子优先置 0。监督导出前，持续出现的人物及天空误检会在同一份面元缓存的 1024 长边网格单遍最近 LAS 深度投影（无累加与法向混合缓冲）上重新核验并置回静态 255（`--no-static-masks` 或 `--no-las-supervision` 时不做，交付的是上述 SAM3 掩码）：
- 人物/影子静态恢复：每个人物/影子连通块（面积不少于 16 个 1024 长边网格像素）取其 LAS 点的世界坐标中位数作为位置，统计能看到该位置（0.8–12 m、在有效域 80% 半径内，不判遮挡）的其他视图中有多大比例在该处被掩码标为 0，及这些视图的曝光时间跨度；比例不低于 0.2 且跨度不低于 40 s、至少 8 幅视图能看到时视为静态（画框、海报里的人物，一直坐着不动的人），块内 0 置回 255。走动的人只在经过时被标，采集操作员只在设备所在处被标，均不满足。静态块 8% 图宽内若有继续被遮的块，它也保持被遮（影子跟随所属人物）。LAS 点少于 12 个或不足块面积 30% 的块、可见视图不足 8 幅的块保持被遮。
- 天空误检与室内识别（配合 `configs/trainer.json` 统一的 `segment_and_ignore`）：
  1. 连通块表面核验：将每幅图未侵蚀的 `128` 区域缩放到 1024 网格后求 8 连通块；面积不少于 16 且包围盒宽高均大于 4 像素的块，用 5×5 椭圆核（零边界）向内腐蚀 2 像素以隔开树梢与屋檐 LAS 飞散点；若腐蚀核仍有至少 16 像素、其中有效 LAS 深度像素不少于 12 个且占比不低于 50%，判为实体表面，将该块外扩 1 格后在原图交 `128` 置回 255。
  2. 场景级封闭室内回退：剔除实体表面块后，统计其余天空像素占该镜头未腐蚀有效域面积不低于 0.5% 的已注册照片数 `images_with_open_sky`；若不足 `min(8, ceil(注册照片数 / 2))` 幅或不足已注册照片的 5%，判为封闭室内场景（`sky.kept = false`），将全部已注册照片掩码中的剩余 `128` 均置回 255，覆盖无 LAS 回波的镜面/亚克力灯箱，并同步恢复该处的 MoGe 深度与法向监督；室外场景（`sky.kept = true`）则保留剩余真实天空 `128` 供训练施加不透明度惩罚，并在原图（按需缩至 2048 长边 ROI）围绕 `128/255` 边界执行原图色彩局部引导滤波细化（`5×5` 腐蚀后的 LAS 实体内核硬锁为 `255`），同时清理紧邻开阔天空且 `9×9` 外环至少 `65%` 为暗色枝叶像素的小面积高亮低饱和枝叶天孔，在 `sky` 中记录 `edge_refined_images`、`edge_to_sky_pixels` 与 `edge_to_static_pixels`。
- 多视角光度校准、过曝裁切硬掩码与连续漫反射权重（v3，`weights/{L,R}/**.JPG.png`）：在 320 长边网格上结合经过 5×5 抑遮挡清洗的 LAS 射线深度，联合执行多视角光度标定、匀光规划、过曝硬屏蔽与连续高光/阴影软降权：
  1. **3 通道 Schur-Complement 光度标定、时序平滑白平衡与有界源头匀光**：融合带法向/入射角筛选的 LAS 表面 0.20 m 体素轨迹（三径向带分层采样）与高置信静态 SfM 3D 点轨迹（`reproj < 2.0 px`、`sRGB 0.12–0.76` 平滑中灰区），按 1 m 世界坐标体素确定性划分 10% 空间块留出集（`holdout`），对 R/G/B 三个物理线性通道独立执行精确 per-track Schur 消元 Huber IRLS（二轮外点剔除 `0.40 EV`）联合求解每镜头 3 阶 KB8 径向暗角多项式（`exp2` 基，物理边界 `[0.40, 1.05]` 与留出集增益校验自动降阶 `3 -> 2 -> 0`）与每帧相对曝光；随后按亮度加权分解每帧 EV，并对两镜头 `log2(R/G)`、`log2(B/G)` 分别做 1D 时序 Huber 平滑（`dt <= 2.5 s`、`±0.80 EV` 限幅、保持中性灰亮度不变）。接着以全场景 EV 的 1D 高斯 KDE 众数 `reference_ev` 为目标，在排除过曝区后的静态像素余量与分布约束下规划安全像素匀光迁移量 `pixel_ev_shift`（提亮不超过 `0.30 EV` 且受 `p98` 线性 `0.92` 高光余量限制；压暗不超过 `0.80 EV` 且满足 `p50 >= 0.30`、`p10 >= 0.06` 及未屏蔽白色过曝像素 `lum >= 0.94 && max_rgb >= 0.975` 占比 `0.5%–2.0%` 线性降额），并执行闭环前向审计将新增白色线性过曝（单通道截断至 `1.0` 后亮度 `>= 0.985`、连通不少于 8 格）与大幅压暗已白色饱和区（`shift > 0.25 EV` 且原图 `lum >= 0.94 && max_rgb >= 0.975`）写入 `post_clip_masks` 并入 `clip_mask`。导出阶段在物理线性空间执行去 3 通道暗角、3 通道白平衡、`-pixel_ev_shift` 匀光及 `0.92` 起双曲软拐点压缩并以 Q=97 4:4:4 RGB JPEG 写出（保留 ICC/XMP 等非 EXIF 段），残余 `residual_ev` 写入 EXIF `APP1`（`manifest.json` 同步记录 `calibrated_ev`、`raw_relative_ev`、`pixel_ev_shift`、`wb_gains`）。
  2. **过曝裁切硬屏蔽 + 多视角 `q=0.42` 连续漫反射权重（保留静态表面深度/法向监督与评测口径）**：先执行过曝裁切核验——在 `<= 25 m` 池中选取至多 32 个更暗共视参考帧校验高光白色饱和区在至少 2 个更暗帧中 `>= 60%` 未裁切，高 EV 室内帧补充远距/无深度窗洞门洞种子并经连通泛洪汇入闭环 `post_clip_masks`（单图 `45%` 熔断），**仅将人物/有效域外/真过曝裁切写入 `masks/` 的 `0`**。对于移动阴影、非过曝镜面高光与 ISP 高光压缩肩部，不再二值挖空 `masks/`（避免破坏静态展板/奖牌/地面的 MoGe 深度与法向监督或产生黑洞），而是换算至全场景 `reference_ev` 统一线性亮度并在 5×5 掩码归一化对数亮度包络下（去除纹理梯度阈值限制，并用全视角原始亮度 `p50` 区分点状高光与静态白字）对至多 32 个 `<= 18 m` 方向分层共视邻域（优先采用处于相机线性响应段 `lum_srgb <= 0.78 && max_srgb <= 0.84` 的参考视角）统计加权 `q=0.42` 漫反射参考亮度，以 Tukey 双权柔和衰减强高光、点状高光、移动/人旁软阴影以及高曝光图内因相机 ISP 高光压缩肩部（`raw_srgb > 0.76` 或 `raw_max > 0.82` 且相对更暗线性邻域明显跌落）造成的暗化偏差（经人物/有效域边界保护的连通去噪核（面积 >= 4）`3×3` 膨胀过渡带与 `3×3` 高斯羽化，光度权重下限 `0.02`，同时保留 `masks/ = 255`），按裁剪后交付视口缩放到最长边 `320` 像素的单通道 `uint8` PNG 导出至 `weights/{L,R}/**.JPG.png`（`manifest.images` 记录 `weight_size`，`request.photometric_normalization.version = 3`）；下游训练在保持法向监督全权重的同时将光度损失、`gradient_residual` 对齐至 `PPISP` 空间，用离线静态权重（不含在线剔除）约束自适应密度误差图并启用延迟对称在线离群保护（`configs/trainer.json` 默认 `"sh_degree": 0`、`"color_variance_weight": 0.05`、`"exposure_correction_grid_start_iter": 3000`）。结果统计写入 `photometric`（`vignetting_*`、`holdout`、`exposure`、`white_balance`、`homogenization`、`clipped_*`、`clip_capped_images`、`diffuse_weights`、`seconds`）。

以上规则与 `sky`、`photometric` 统计一并写入 `cache/static_masks.json`。因此同一画面在不同视图中不会被一部分遮、一部分不遮，交付的掩码和深度、法向监督用同一份掩码。SfM 只提取 255 区域；有效域按初始标定冻结。

深度和法向监督默认在导出阶段生成，采用 LAS 尺度对齐的 MoGe-2 SSI 监督（`source = "moge2_ssi"`、`depth_source = "moge2_ssi"`、`depth_alignment = "las_face_bilateral_v1"`、`depth_encoding = "linear_[min_d,max_d]_to_[0.02,1.0]*65535"`、`normal_source = "moge2_normals"`）；`run` 与独立 `export` 均可用 `--no-las-supervision` 关闭监督（并跳过静态掩码核验）。监督图默认最长边 1024（`--supervision-max-size`，128–4096），不放大原图，相机内参按输出尺寸缩放。深度是尺度偏移不变（SSI）沿视线距离，uint16 灰度 PNG（`depth_format = "uint16_png_ssi"`、`depth_type = "ray_distance"`），0 表示无效，有效像素按 `[0.1%, 99.9%]` 抗极端离群后的 `[min_d, max_d]` 线性映射到 `[0.02, 1.0] * 65535`（`[1311, 65535]`）。法向是 OpenCV 相机坐标下朝向相机的单位向量（`normal_frame = "opencv_camera"`），RGB 为 `round(127.5*(n+1))`，`(0, 0, 0)` 表示无效。定制 LFS 读取 `supervision.json` 后自动关闭内置单目重复推理并将相机法向空间设为 `camera-opencv`；当命中 `depth_alignment = "las_*"` 与 `depth_encoding = "linear_[min_d,max_d]_to_[0.02,1.0]*65535"` 时，直接按每帧 `[min_distance_m, max_distance_m]` 构造精确米制锚点跳过 RANSAC 拟合，并在 3DGUT 路径启用边界掩码死区 Huber 米制深度、无量纲相对深度失真与 5×5 保守 Free-Space 抑制（非 LAS 对齐包仍保持原 `ssi` RANSAC 通路）；沿视线深度须用其 GUT 光栅化（`--gut`）训练；图像仍保留原始鱼眼畸变。

静态掩码核验与监督预投影共用单遍最近 LAS 面元深度投影（不读取初始化 PLY，不单独输出原始 LAS 深度/法向图）：在 LAS 扫描期按已注册相机中心包络向外扩 `65.535 m + 2.5 m` 预裁剪超出任何相机有效射程的点，计算切平面面元后以 50 万记录分块在 GPU 投射最近射线距离；完成静态掩码后复用同一面元缓存投影交付相机的 LAS 毫米深度临时缓存，随即释放 LAS 缓存再启动 MoGe-2 推理。`--supervision-cache-mb` 设置该 LAS 面元缓存的 GPU 上限（默认 4096 MiB，1–8192），实际预算不超过空闲显存的四分之一，超出预算时以紧凑文件双缓冲上传。

MoGe-2 监督导出流程：每个镜头按监督图相机规划一组 644×644 的方形 90° 透视子图（160° 鱼眼通常为 1 张前向加一圈倾斜 50° 共 6–7 张，每条有效视线离某张子图边缘至少 5°）；`lenses` 的 `plan` 记录该组规划。原图先在 CPU 按面积缩小到子图中心像素密度的 2 倍，再在 GPU 核函数中双线性重采样到各子图；卷帘快门交付下按每幅照片的读出转角 `ω` 对每个子图像素做 3 步不动点去卷帘反向映射，直接抽取对齐参考姿态 `s = 0` 的透视子图消除转身切线错位。MoGe-2 按引擎支持的最大动态批大小（默认 `B = 1..8`，低显存收敛至 `<= 4`）分批推理，在 64×64 网格上按已知焦距恢复 z 偏移并换算成沿视线距离与子图法向，随后各子图对经过 5×5 局部最小抑遮挡穿透清洗后的 LAS 深度做稳健尺度和必要时的 Cauchy IRLS 仿射对齐（至少 256 个高权重样本），缺乏足够 LAS 支撑的子图回退重叠区中位比值，加权混合射线距离与单位法向，并在 `[log d, x, y]` 三维双边网格上对对数残差做保边平滑校准（剔除玻璃穿透与严重离群点），在卷帘快门下将融合法向旋回像素读出线相机系（`normal_estimator` 记为 `moge2_faces_readout_line`，全局快门为 `moge2_faces`），确保法向朝向相机。

最终应用静态掩码（仅保留 `255`，排除天空 `128`、动态人物及有效域外 `0`）、可选 `--supervision-mask-dir` 排除掩码以及 `--supervision-boundary`（默认 4 个监督图像素，1–32）膨胀边界屏蔽区：屏蔽区及 MoGe-2 无效像素在 `depths/` 置 `0`、在 `normals/` 置 `(0, 0, 0)`。原照片及 RGB 掩码保持不变；整包须同时具有有效深度与有效法向像素。几何引擎取自 `--supervision-models` 的模型包；未指定时 `run` 用 `--models`，否则用默认模型目录，缺少 `geometry` 引擎时在解析参数阶段即报错。

`supervision.json` 记录 `source = "moge2_ssi"`、`depth_source = "moge2_ssi"`、`depth_alignment = "las_face_bilateral_v1"`、`depth_encoding = "linear_[min_d,max_d]_to_[0.02,1.0]*65535"`、`normal_source = "moge2_normals"`、`depth_format = "uint16_png_ssi"`、`depth_type = "ray_distance"`、`normal_format = "rgb8_png"`、`normal_frame = "opencv_camera"`、`normal_estimator`、`semantic_margin_pixels`、`photometric_weights`、`geometry_model`（模型包、引擎 SHA256 与文件身份、来源、子图尺寸、token 数）、`lenses`、逐图字节数/有效像素/距离范围/分面尺度和偏移及 `las_alignment` 统计、`inference` 与 `timing`；请求身份包含 `max_size`、`boundary_pixels`、`rolling_shutter`、`depth_alignment`、`las_file`、几何引擎身份、静态掩码与额外排除掩码元数据（以及裁剪窗口）。

manifest 的 request.supervision 记录监督请求身份。supervision 字段与 `supervision.json` 相同，保存按已注册照片原顺序排列的尺寸、文件字节数及有效像素数量。`check` 逐图回读，核对 `supervision.json` 合同字段、类型、尺寸、身份、有效像素数量、SSI 深度编码范围（非零像素 `>= 1311`）、法向单位长度及其对应有效深度（`has_normal => has_depth`）。关闭监督时不生成这些字段和文件。

manifest 保存请求、已注册图像身份、文件大小、初始化参数；启用静态掩码时另含 `static_masks`（保存 `cache/static_masks.json` 中除逐连通块明细 `component_rows` 之外的统计与规则摘要，包括 `sky` 字段）。卷帘校正被采用时 request 另含 `rolling_shutter`：`cache/shutter.json` 的文件身份与 `photos: "original"`。图像条目的 image_index 是原输入的 0 起始索引。quality 保存所选模型拟合验收和限值。省略 PLY 时不生成 initialization.json，清单中的 initialization 为 null。

manifest 和 quality 的 registration 保存 complete、selected_images、registered_images，以及 unregistered_images 名单（image_index、name、0 起始 camera）。默认首次停滞就处理已注册部分；可配置 `sfm_repair_rounds=1` 或 `2`，在补匹配轮数用完后处理已注册部分。子集收尾还会排除末尾标定可靠支撑不足的曝光，其原始索引和支撑记录保存在 sfm.calibration_selection。只有通过支撑检查的已注册照片参与后续优化、几何验收及配准；原有验收门限不变。quality.images 保留全部输入的顺序与 registered 标记，未注册条目的观测数为 0、误差统计为空。quality.passed 表示已重建部分通过验收，不表示全部照片注册成功；check 同时返回 registration。

quality.calibration 保存两镜头的初始与最终内参、主点偏移，以及每 20° 入射角区间的重投影误差。角度按初始标定计算，便于比较镜头中心与边缘；这些统计是拟合诊断，不代表独立精度。

quality.track_lengths 保存有效三维点的轨迹长度数量、中位数和 P90，tracks_at_least_3、tracks_at_least_8 保存至少 3、8 次观测的点数，便于结合每图观测数及 occupied_cells 比较配对策略的覆盖。

`result.json` 的 joint 保存联合优化候选的完整质量、点位移统计、逐图退化明细、各项接受条件及 failed_checks；候选被拒绝时保留所选原模型，验收阈值不因诊断而放宽。

默认在 LAS 对齐后执行卷帘快门校正和独立稀疏点清理，再导出最终所选模型。`result.json` 的 alignment、joint、sfm 是各阶段原报告；shutter 保存读出时间、读出转动与平移统计、`imu_attitude`（含曝光–IMU 时钟偏置自标定 `time_offset`）、各镜头读出一致性、留出误差、同组 LAS 平面误差、观测换算位移、各项检查及 used；cleanup 单独保存稀疏点清理与验收。配置 `"rolling_shutter": false` 时 shutter.enabled=false。quality 与 navigation_error_m 更新为最终模型统计。

清理候选未通过时保留清理前的模型，未采用原因明确记录。允许清理最多 1% 的已注册照片，且每镜头至少保留 98%、可靠 LAS 支撑轨迹至少保留 90%；最终照片名单以交付 registration 为准。清理不改变相机内参和位姿，也不改变 LAS 初始化 PLY。

同级 `.pending` 中完成写入及回读后重命名发布。同一请求的完整包可复用，不同请求拒绝覆盖。`check --output` 只依赖交付包，检查结构、引用、尺寸、校准 EXIF EV（若 `manifest.images` 含 `calibrated_ev` 则核验回读误差 `<= 0.01 EV`）、PLY 和几何拟合，不代表独立精度或渲染效果。

`manifest.json` 的 `project` 记录选中的 PCC `.pro` 文件名。`check --output DIR --project FILE.pro`（可给路径，按文件名比较）同时核对交付包是否来自指定解算项目。

格式依据：[COLMAP output format](https://colmap.github.io/format.html)。

## 卷帘快门交付

卷帘校正被采用的模型交付原图与原掩码（默认按有效域无损裁剪），特征坐标为交付图像像素，位姿为参考位姿；`rolling_shutter.json` 给出每幅照片的读出运动。按全局快门读取时视图间仍有卷帘偏差，下游须使用该文件（定制 LFS 在 `--gut` 且未开启 undistort 时按卷帘相机训练，否则告警并按全局快门处理）。照片不做重采样：读出平移的像素偏移随深度变化，按 LAS 距离重采样会在近处动态物体和 LAS 空洞处错位。points3D.bin 的 error、quality.json 与 `check` 均按卷帘投影计算；监督导出的 90° 透视子图先按读出转角去卷帘采样至参考姿态 `s = 0`，融合后的法向再旋回各像素自身的读出线相机系。supervision.json 的 projection 为 `rolling_shutter_readout_line`，normal_estimator 为 `moge2_faces_readout_line`。

`rolling_shutter.json` 是卷帘交付的相机接口：

```text
readout_axis   "rows" 或 "columns"，两镜头共用
cameras        [{camera_id, readout_s}]，readout_s 为整帧读出时间（秒），符号表示读出先后
images         [{image_id, camera_id, name, rotation_rad[3], velocity_m_s[3], translation_m[3]}]，与 images.bin 的 ID 一致；读取时 camera_id 可省略，存在即须与 images.bin 相符
```

读出比例 `s = (c - N/2)/N`，c 为像素沿读出轴的坐标（左上像素中心为 0.5），N 为该轴的图像尺寸。参考位姿为 images.bin 的 world-to-camera `R, t`（s = 0）。三维点 X 在读出比例 s 处的相机坐标为 `p(s) = Exp(s[rotation_rad]×)(R X + t - s·translation_m)`，`translation_m = readout_s · velocity_m_s`；rotation_rad、velocity_m_s、translation_m 均在 OpenCV 参考相机坐标中，rotation_rad 与 translation_m 对应整帧读出（每轴角分量优化上界 `0.10` rad）。X 的像素 x 满足 `x = fisheye(p(s(x)))`，从 s = 0 起不动点迭代 3–4 次即收敛；同文件 `model` 字段为 `"SO(3) rolling shutter about the reference pose"`，`result.json` 中 `shutter.imu_attitude.time_offset` 记录曝光–IMU 时钟偏置自标定详情。

## 有效域裁剪交付

`run` 与 `export` 默认把每个镜头的照片、掩码与模型裁到该镜头有效域的外接窗口；`--no-crop-valid-domain` 交付整幅原图。裁剪使下游少处理鱼眼圆外恒为 0 的像素。窗口取该镜头全部已注册照片静态掩码非零像素的外接框，四周各留 16 px（不超出原图）；左上角对齐照片 JPEG 的 MCU 网格。关闭静态掩码或监督时，照片按 DCT 系数无损裁剪、不重新编码并原样保留 EXIF（EXIF 中记录的像素尺寸不改写）；默认启用静态掩码且全局光度校准通过时，在裁剪窗口内于物理线性空间完成去暗角、白平衡、安全匀光与 `0.92` 软拐点压缩并以 Q=97 4:4:4 RGB JPEG 写出（保留 ICC/XMP 等非 EXIF 段并将残余相对曝光写入 EXIF `APP1`）。掩码按窗口精确裁剪，`weights/` 同步按窗口裁切并重采样至 320 长边。窗口沿读出轴以图像中心对称，读出比例只按 `N'/N` 缩放（N、N' 为裁剪前后沿读出轴的尺寸）：`rolling_shutter.json` 的 `rotation_rad`、`translation_m` 与 `readout_s` 同乘 `N'/N`，`velocity_m_s` 与参考位姿不变，交付在裁剪后图像上按同一公式 `s = (c - N/2)/N` 使用。

`cameras.bin` 的宽高为窗口尺寸，主点减去窗口原点，畸变参数不变；`images.bin` 的特征坐标减去窗口原点，`points3D.bin` 的 error 按裁剪后的投影重新计算。深度、法向监督按裁剪后的相机生成，静态及额外排除掩码先裁到窗口再缩放到监督图；监督图像素只有在其原图面积内全部有效时才有效（对无效像素做面积缩放，结果须严格为 0）。`manifest.json` 的 request 另含 `crop`（`mode: "valid_domain"` 与两镜头窗口），监督请求另含 `crop_windows`，因此裁剪与未裁剪交付不会被视为同一请求。`check` 对裁剪交付按同样规则回读。

## 下游训练与评测核验契约

结合下游默认训练配置 [trainer.json](../configs/trainer.json)，评测与交付核验须遵守以下三项契约：

1. **全视角评测与数据集切分（`eval_all`）**：
   - `configs/trainer.json` 默认 `"eval_all": false`、`"test_every": 8`（训练期按每 8 张抽样 1 张作为 `test` 留出集，评估留出视角泛化；开启评测时按 `eval_steps` 执行）。
   - 执行离线全视角评测或全量拟合复核时，须显式开启 `--eval-all`（等价 `"eval_all": true`）：全部已注册视角同时用于训练与评测，`per_image_metrics.json` 中各视图 `"split"` 统一记为 `"train"`，不保留 `test` 留出集，也不对留出相机执行额外姿态重拟合；因此 `--eval-all` 指标反映全场景拟合度，不代表留出泛化精度。
2. **静态前景计分（`mask == 255`，`segment_and_ignore`）**：
   - 交付 `masks/` 与 `configs/trainer.json` 默认 `"mask_mode": "segment_and_ignore"` 约定统一的三段单通道语义：`255` 为静态前景、`128` 为开放天空（`segment`，训练期受不透明度惩罚并配合 `"bg_color": [0.0, 0.0, 0.0]` 收敛为透明黑底）、`0` 为动态人物及有效域外（`ignore`）。
   - 主评测指标（`PSNR`、`SSIM`、`LPIPS`、`FLIP` 及前景色偏 `bias` / `bias_corr`）在 `segment_and_ignore` 下**只对 `mask == 255` 像素计分**（内部按 uint8 `> 250` 或浮点 `> kMaskKeepMin` 分类 keep 掩码），`128` 与 `0` 均不计入前景指标；如需分项查看全图或被遮挡区表现，须配合 `--eval-mask-report mask,unmasked,inverted` 输出对照集合。
   - 该计分口径在所有掩码来源上必须一致：数据集内建 sidecar `masks/`、RGBA `alpha`（`"use_alpha_as_mask": true`）以及离线显式 `--eval-mask masks:<folder>` 在 `segment_and_ignore` 模式下均不得退化为 `mask_threshold = 0.5` 的普通二值化（避免将 `128/255` 天空误纳入 keep，导致过曝天空原图与透明黑底渲染计算 MSE 而引起显著负色偏与 PSNR 失真）。
3. **卷帘快门日志与指标核验**：
   - 当交付包存在 `rolling_shutter.json` 时，下游定制 LFS 须在初始化与每次评测输出中显式可核验卷帘启用状态：
     - **初始化日志**：满足 `"gut": true` 且 `"undistort": false` 时记录 `Rolling shutter: <N> cameras render with the {row|column} readout motion from rolling_shutter.json`；若条件不满足则明确告警并按全局快门处理。
     - **评测输出**：每次有效评测统计实际激活卷帘渲染的相机数 `rolling_shutter_cameras`（要求 GUT 启用、相机附带卷帘参数、非 `undistort` 针孔网格、非 `EQUIRECTANGULAR` 且读出轴渲染尺寸 `> 1`，并在 `"undistort": true` 且 `"eval_space": "distorted"` 原生 GUT 评测路径保持透传），同时写入运行日志（`Eval: rolling shutter active for <K> / <N> cameras` 与汇总 `EvalMetrics::to_string()` 中的 `rolling_shutter_cameras: K/N`）及 `metrics_report.txt` 中的 `Rolling shutter cameras: K / N`（`metrics.csv` 与 `per_image_metrics.json` 保持既有结构不变）。
     - **核验准则**：按 `configs/trainer.json` 默认口径（`"gut": true`、`"undistort": false`、`"eval_space": "distorted"`）评测含 `rolling_shutter.json` 的交付包时，须核验 `K == N > 0`；未附带 `rolling_shutter.json`、关闭 `--gut` 或在去畸变空间评测时则应为 `K == 0`。

## 自适应训练配置（`trainer.json`）

`run` 与 `export` 默认在导出阶段基于模板 [trainer.json](../configs/trainer.json)（可用 `--trainer-config JSON` 覆盖模板，内嵌副本随二进制编译）结合当前场景统计与本机 GPU 总显存（可用 `--vram-gb GB` 覆盖）自适应生成 `<output>/trainer.json`，`Geosun.exe` 训练时优先加载该文件，不存在时回退公共模板：

- **训练图像数 `N_train`**：由已注册图像数扣除 `test_every` 留出测试集得到（当 `eval_all: true` 或 `test_every <= 1` 时取全部已注册图像）。
- **迭代步数与调度同步缩放**：以 `550` 张训练图对应模板 `base_iters`（默认 `30000`）为锚点按幂律 `round_1000(clamp(base_iters * (N_train / 550)^0.6, 20000, 120000))` 推导 `iterations`，并按同一比例同步缩放 `start_refine`、`stop_refine`、`reset_every`、`refine_every`、`morton_reorder_interval`、`sh_degree_interval`、`grow_until_iter`、`color_variance_warmup_iters`、`exposure_correction_grid_start_iter`、`eval_steps` 与 `save_steps`，同时将 `steps_scaler`、`image_count_scaler`、`image_count_scaler_total` 保持为 `1.0` 避免下游二次缩放。
- **高斯上限 `max_cap` 与初始化点云封顶**：综合几何表面预算 `cap_geom = 3.0 * geom_points`（优先取 `roi_voxel_points_before_cap`，其次 `voxel_points_before_cap` 或 `points`）、视角支撑上限 `cap_views = 8000 * N_train` 与显存硬上限 `cap_vram = round_100k(clamp((vram_gb - 2.5) * 1800000, 4000000, 24000000))`，取 `max_cap = round_100k(clamp(min(cap_geom, cap_views), 1500000, cap_vram))`（下游通过模板的 `grow_fraction = -1`、`shs_lr = -1` 自动跟随 `max_cap` 联动）；`points3D.ply` 总点数封顶联动为 `min(2500000, max(100000, round(0.75 * max_cap)))`。
- **存量交付补生成与校验**：对于未包含 `trainer.json` 的已有交付包，可执行 `geosunpre trainer-config --output DIR [--vram-gb GB] [--trainer-config JSON]` 从 `manifest.json` 直接生成；新交付包会在 `manifest.json` 的 `trainer_config`（及 `request.trainer_config`）记录摘要，由 `check` 核验文件尺寸与合法性。
