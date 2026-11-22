# COLMAP 与 PLY 交付

交付直接位于 `--output` 根目录，下游定制 LFS 训练的数据路径即为该目录。

```text
OUTPUT/
  images/{L,R}/**.JPG
  masks/{L,R}/**.JPG.png
  depths/{L,R}/**.JPG.png       # 默认生成，--no-las-supervision 省略
  normals/{L,R}/**.JPG.png      # 默认生成，--no-las-supervision 省略
  sparse/0/{cameras,images,points3D,rigs,frames}.bin
  sparse/0/points3D.ply
  roi.json                    # 默认随 PLY 生成，--no-roi 或 --no-ply 时省略
  manifest.json
  quality.json
  initialization.json
  supervision.json            # 默认生成，--no-las-supervision 省略
  # masks 默认为持续人物及表面/室内天空误检改回静态后的掩码，--no-static-masks 时为 SAM3 掩码
  rolling_shutter.json        # 卷帘校正被采用时生成
  result.json                 # 各阶段报告，与工作目录 result.json 相同，不参与 check
```

COLMAP 小端二进制，`OPENCV_FISHEYE` 模型编号 5、8 个参数。相机 ID 1、2；每幅图像独立 frame，使用镜头对应的一传感器 rig。内部索引从 0 开始；仅导出已注册照片，按原输入顺序将图像/frame ID 连续重编号为 1 到注册数量，轨迹中的图像引用同步转换，内部缓存编号保持不变。

位姿 world-to-camera，COLMAP 四元数顺序 `w x y z`。图像名相对于 images，保留原图名、鱼眼畸变与全部特征编号。照片和掩码不做重采样：默认无损裁到各镜头有效域窗口（见“有效域裁剪交付”），`--no-crop-valid-domain` 时为整幅输入照片拷贝；来源为 `--images` 所给目录中的照片。图像名统一为 `镜头/文件名`（如 `L/xxx.JPG`）。卷帘快门校正被采用时，位姿为读出中线时刻的参考位姿，并附 `rolling_shutter.json`，见下节。点轨迹与图像引用相互一致，无三维点的特征用 uint64 最大值。下游须支持原生鱼眼模型。

points3D.bin 的 error 为最终所选模型在该点全部观测上的平均重投影距离，单位像素；联合优化后重新计算，交付回读时验证。

相机和稀疏点已在局部 LAS 坐标（局部 LAS 即输入坐标，ECEF LAS 为其转换后的导航 ENH 局部坐标，见[输入约定](input.md)），下游不再应用配准变换。`points3D.bin` 是清理并验证后的视觉稀疏点及轨迹（保持纯 LAS 锚定口径）；`points3D.ply` 主体保留已注册相机 50 m 视锥范围内的有效 LAS 点，按到最近相机的距离分档体素均值采样（10 m 内 1×、20 m 内 2×、40 m 内 4×、50 m 内 8× 基准体素；基准体素默认 0.05 m），并从清理前视觉模型补充距已注册图像 0.25–250 m、0.25 m 内无有效 LAS 覆盖的高置信静态多视图视觉点（不少于 3 视图、均值重投影不超过 1.5 px、视差至少 2°、无 0 掩码中心且至少 75% 且不少于 3 个视图的 5×5 邻域全为 255，按至少 0.10 m 网格去重并限制补充最多 10 万点），LAS 随同调整上限使 PLY 总点仍封顶 250 万；当启用 `roi.json` 且体素点超出该上限需要抽稀时，外围最多保留 5%（不超过 12.5 万点）作背景支撑，其余优先分配给 ROI 内部点，`initialization.json` 另记录 `las_points`、`visual_supplement_points`（启用 ROI 时另记录 `roi_las_points` 与 `outside_roi_las_points`）。文件为二进制小端，每点 float32 XYZ、零法向量 XYZ、uint8 RGB，共 27 字节。PLY 与相机同一坐标，`initialization.coordinate_frame` 为 `input_las_xyz` 或 `ecef_to_navigation_enh`。LAS 头部范围（ECEF LAS 为转换后范围）的坐标绝对值须不超过 16777 m，使 float32 舍入不超过 1 mm，否则明确报错。下游需明确选择此 PLY 初始化 3DGS。

`roi.json` 默认随初始化 PLY 在 `export` 阶段自动生成（`--no-roi` 或 `--no-ply` 时省略）：基于 50 m 视锥内有效 LAS 在 1 m×1 m XY 格网的原始点占据（单格不少于 30 点），结合全部已注册相机轨迹连通光栅，经 2 m 圆形膨胀并从相机种子做 4 邻域连通筛选后提取外轮廓（多块时取外凸包），迭代 `approxPolyDP` 简化为 3–64 顶点的简单非自交 2D 多边形（兜底外切凸包或包围框），保证 100% 已注册相机严格落入内部；Z 范围取多边形内体素 LAS 分布与相机高度的稳健包络（`z_min = min(p001, cam_z_min - 2) - 1`，`z_max = min(max(p999, cam_z_max) + 15, max(p999 + 3, las_max + 1, cam_z_max + 3))`）。文件记录 `version: 1`、`type: "polygon_prism"`、`coordinate_frame`、`z_min`/`z_max`、`vertices_xy`（工作坐标二维顶点）、`bounds_min`/`bounds_max`、`params` 与 `stats`；定制 LFS 加载数据集时自动挂载为训练模型 `CropPolygon`，配合 [trainer.json](../configs/trainer.json) 在训练期降权外围并默认终态 `--crop` 裁切输出。`manifest.json` 的 `request.roi` 与顶层 `roi` 记录该文件身份，`check` 回读校验文件尺寸、多边形几何合法性与 100% 相机包含性。

掩码为与交付照片同尺寸的单通道 PNG：静态 255、天空 128、人物及有效域外 0。SAM3 grounding 在整幅照片上决定人物与天空：person、shadow、robot、sky 检测分数 ≥ 0.5 时有效；人物检测有一半以上被分数不低于它的机器人检测覆盖时视为人形展品，保持静态；影子在 288 格掩码上须与保留人物相距不超过 24 格，且若单个影子检测面积超过整格网的 8% 或（超过 3% 且）超过接触人物连通块总面积的 3 倍，仅裁剪保留距人物 24 格内的局部区域，防止行人踩入建筑大阴影后连带抹掉整片静态地面（并在 `masks_profile.json` 记录 `shadows_clipped`）；远离人物的家具和展品阴影保持静态；人物与影子的并集外扩 1 格后放大到原图。天空为 sky 检测掩码的并集，不外扩，放大到原图后在有效域内置 128，人物与影子优先置 0。监督导出前，持续出现的人物及天空误检会在同一份面元缓存的 1024 长边网格单遍最近 LAS 深度投影（无累加与法向混合缓冲）上重新核验并置回静态 255（`--no-static-masks` 或 `--no-las-supervision` 时不做，交付的是上述 SAM3 掩码）：
- 人物/影子静态恢复：每个人物/影子连通块（面积不少于 16 个 1024 长边网格像素）取其 LAS 点的世界坐标中位数作为位置，统计能看到该位置（0.8–12 m、在有效域 80% 半径内，不判遮挡）的其他视图中有多大比例在该处被掩码标为 0，及这些视图的曝光时间跨度；比例不低于 0.2 且跨度不低于 40 s、至少 8 幅视图能看到时视为静态（画框、海报里的人物，一直坐着不动的人），块内 0 置回 255。走动的人只在经过时被标，采集操作员只在设备所在处被标，均不满足。静态块 8% 图宽内若有继续被遮的块，它也保持被遮（影子跟随所属人物）。LAS 点少于 12 个或不足块面积 30% 的块、可见视图不足 8 幅的块保持被遮。
- 天空误检与室内识别（配合 `configs/trainer.json` 统一的 `segment_and_ignore`）：
  1. 连通块表面核验：将每幅图未侵蚀的 `128` 区域缩放到 1024 网格后求 8 连通块；面积不少于 16 且包围盒宽高均大于 4 像素的块，用 5×5 椭圆核（零边界）向内腐蚀 2 像素以隔开树梢与屋檐 LAS 飞散点；若腐蚀核仍有至少 16 像素、其中有效 LAS 深度像素不少于 12 个且占比不低于 50%，判为实体表面，将该块外扩 1 格后在原图交 `128` 置回 255。
  2. 场景级封闭室内回退：剔除实体表面块后，统计其余天空像素占该镜头未腐蚀有效域面积不低于 0.5% 的已注册照片数 `images_with_open_sky`；若不足 `min(8, ceil(注册照片数 / 2))` 幅或不足已注册照片的 5%，判为封闭室内场景（`sky.kept = false`），将全部已注册照片掩码中的剩余 `128` 均置回 255，覆盖无 LAS 回波的镜面/亚克力灯箱，并同步恢复该处的 MoGe 深度与法向监督；室外场景（`sky.kept = true`）则保留剩余真实天空 `128` 供训练施加不透明度惩罚，并在原图（按需缩至 2048 长边 ROI）围绕 `128/255` 边界执行原图色彩局部引导滤波细化（`5×5` 腐蚀后的 LAS 实体内核硬锁为 `255`），同时清理紧邻开阔天空且 `9×9` 外环至少 `65%` 为暗色枝叶像素的小面积高亮低饱和枝叶天孔，在 `sky` 中记录 `edge_refined_images`、`edge_to_sky_pixels` 与 `edge_to_static_pixels`。

以上规则与 `sky` 统计一并写入 `cache/static_masks.json`。因此同一画面在不同视图中不会被一部分遮、一部分不遮，交付的掩码和深度、法向监督用同一份掩码。SfM 只提取 255 区域；有效域按初始标定冻结。

深度和法向监督默认在导出阶段生成，默认来源为下节的 MoGe 单目监督；`--supervision-source las` 选用本节的 LAS 投影监督。`run` 与独立 `export` 均可用 `--no-las-supervision` 关闭全部监督。监督使用最终所选相机和原始有效 LAS：先在点云上为每点估计法向与面元足迹，再以 50 万记录分块、GPU 有界批次投影面元，位姿与投影为双精度；不读取初始化 PLY。监督图默认最长边 1024，不放大原图，相机内参按输出尺寸缩放。深度是沿视线距离，uint16 毫米，0 表示无效，0.05 m 以下及 65.535 m 以上留空。法向是 OpenCV 相机坐标下朝向相机的单位向量，RGB 为 `round(127.5*(n+1))`，黑色无效。定制 LFS 按 `supervision.json` 的 `depth_type`、`depth_format`、`normal_frame` 自动选择米制深度和相机坐标法向，沿视线深度须用其 GUT 光栅化（`--gut`）训练；图像仍保留原始鱼眼畸变。

导出流程在一次 LAS 扫描中为静态掩码分析与监督图导出建立并复用同一份面元缓存（跳过或已有有效静态掩码缓存时延迟至监督导出时构建）。提供已注册相机模型时，在 LAS 扫描期按已注册相机中心包络向外扩 `65.535 m + 2.5 m` 预裁剪超出任何相机有效射程的点（`point_cache.culled_out_of_reach`）。保留的整数 XYZ 写入工作缓存目录的紧凑坐标文件，同时沿有效范围内的最长轴统计 65536 格直方图。法向估计按该轴分成若干片，每片连同两侧 15 cm 的邻域点不超过开始时空闲显存一半可容纳的点数（至多 6400 万），单片时整片一次处理；每片结果为每点 20 字节的面元（整数 XYZ、八面体编码的世界法向、足迹半径），每 50 万点切片并记录其原始整数坐标包围盒，之后删除坐标文件。面元总量能放入预算时一次上传后在 GPU 复用；否则写入紧凑面元文件，按相机批次双缓冲读取、上传与投影（GPU 常驻与文件流式模式均按当前批次相机射程包围盒跳过无重叠的面元分块），完成后删除。`--supervision-cache-mb` 设置 GPU 面元缓存上限（默认 4096 MiB，1–8192），实际预算还限制为空闲显存的四分之一；深度图等工作缓冲单独分配。使用文件缓存时，主机和 GPU 传输缓冲各固定约 20 MB；法向分片的主机缓冲以片容量为界，内存不随 LAS 点数无界增长。静态掩码读取缩放、监督图 PNG 写出及交付回读与 GPU 并行，最多各 6 幅在途。`supervision.json` 的 `point_cache` 记录存储方式、扫描次数、预裁剪点数、分块投射/跳过计数、面元字节数、上传量和准备耗时，`surfels` 记录法向估计参数、分片数、有法向点数、足迹与平面均方根的中位数及耗时。缓存预算仅控制执行资源，不改变监督请求身份或像素含义。面元混合在 GPU 上以浮点原子加累加权重、深度和法向，累加顺序随线程调度变化，因此相同输入与设置重跑或改变缓存方式时监督图不逐字节一致：生产车间实测约十万分之一的有效深度像素相差 1 mm（个别像素可达约 1 cm），约千万分之二的有效法向像素相差一个 8 位量化级（约 0.45°），另有十余个像素的深度或法向有效性翻转。`check` 只回读交付包自身记录的有效像素数，不与重跑结果逐字节比较。

点云法向：以 3 cm 哈希网格在 6 cm 范围内求每点的 16 个最近邻，第 16 近邻距离为点间距（不足 16 个时取 6 cm）；不足 10 个近邻的点没有法向。法向取以该点为球心、半径为点间距与 3 cm 中较大者（不超过 6 cm）的球内点做 PCA：第一次全部点，第二次只保留距平面 max(2 cm, 2.5σ) 内的点，第三次收敛到 2 cm 内点带。有效平面须至少 10 个内点、内点占球内点至少 60%、法向方向的均方根 σ 不超过 1 cm、次特征值不小于最大特征值的 5%（排除线状支撑），且该点距平面不超过 2 cm；点间距还不得超过其近邻点间距中位数的 2 倍，表面前方的零散回波因近邻都在远处的密集表面上而被排除。有法向且 16 个近邻都在范围内的点得到足迹圆盘，半径为 2 × min(点间距, 近邻点间距中位数) / 4，至多 5 cm；其他点只投影到自身像素，孤立回波仍在投影后被剔除。

`--supervision-source las` 的面元投影分两遍：投影核函数在参考相机坐标先以最大射程与最大 160° 视场保守角包络（含卷帘读出平移与旋转余量）提前剔除视锥外点，再对其余点执行 3 步卷帘读出线不动点迭代；第一遍求每个像素最近的视线距离，面元覆盖视线与其切平面交点落在圆盘内的像素（离中心不超过 8 个监督图像素，交点视线与法向夹角余弦须大于 0.05），点自身所在像素总被覆盖；第二遍把距最近距离 1 cm 加 1% 以内的全部面元按 `1-(偏离/半径)²`（下限 0.05）加权平均，得到该像素的深度和法向，法向在相机坐标中朝向相机，有法向面元的权重须占一半以上，且单位法向的加权平均长度不小于 0.95，否则该像素只有深度。静态掩码分析仅使用第一遍最近距离投影（不分配累加混合或法向缓冲）。像素坐标、视线及卷帘读出线与此前相同。

投影后先剔除缺少局部深度支撑的孤立点（规则同下述深度有效性），再剔除近处表面空隙中漏出的远点，最后以充分且深度一致的邻居补小孔。孤立异常点因此既不挖空周围表面，也不触发断层。遮挡断层取两侧均有支撑的台阶：5×5 邻域内背景比前景深出前景深度的 10% 加 5 cm；斜看的地面、天花板和侧墙相邻像素间的深度渐变不构成断层。遮挡断层两侧的余量不对称：背景侧只去掉台阶处前景 2 像素内的背景像素；前景侧去掉位于这些台阶像素前方、距离不超过 `ceil(1 cm × fx / 深度)` 像素（至多 24）的前景像素，fx 为监督图焦距。LAS 前景轮廓因横向测距噪声和多次扫描叠加，比照片中的轮廓向外扩出约 1 cm，前景侧余量覆盖这部分；在室内场景实测，4 m 内照片边缘与保留的 LAS 前景边缘相差 0–1.3 像素，整体平移小于 1 像素。语义掩码边界（静态掩码中的人物、天空及有效域边界）默认留出 4 个监督图像素，由 `--supervision-boundary` 设置。深度和法向分别判断有效性：深度在 5×5 邻域中须有至少 4 个深度一致的像素（含自身，距离差不超过中心深度的 2% 加 1 cm），不依赖法向是否可求。法向取上述混合面元法向，只在该像素保留投影所得深度（非补孔）时有效，且法向与视线夹角余弦绝对值至少 0.3。法向有效时深度必须有效，黑色法向不表示对应深度无效。缺测允许单图完全为空，但整包必须有有效深度和法向。以上是几何一致性筛选，不证明真实可见性或自动识别玻璃；`--supervision-mask-dir` 可按原图身份追加仅影响几何的排除掩码，原 RGB 掩码不因 LAS 缺测而改变。

### MoGe 单目监督（解耦模式）

默认的 `--supervision-source moge` 走解耦混合监督（`source = "las_cleaned_moge2_normals"`）：`depths/` 保留经有效 LAS 投影与逆深度反向清洗后的真实毫米级 LAS 深度（`depth_source = "las_moge_cleaned"`），`normals/` 则在非边缘且通过 RANSAC 清洗的实体平面上优先写入多视角几何一致的 LAS 面元法向，仅在 LAS 缺测、边缘带及非平面细节处回填 MoGe-2 ViT-L normal 估计并融合的密集表面法向（`normal_source = "las_planar_moge2_fill"`），两者均不在 2D 空间硬耦合扭曲。文件、编码、尺寸与上述 LAS 监督相同。几何引擎取自 `--supervision-models` 的模型包；未指定时 `run` 用 `--models`，否则用默认模型目录。它与工作目录绑定的重建模型包无关，已有工作目录可直接 `export`。缺少 geometry 引擎时在解析参数时即报错，不等到导出。

每个镜头按监督图相机规划一组 644×644 的方形 90° 透视子图（160° 鱼眼为 1 张前向加一圈倾斜 50° 共 7 张，每条有效视线离某张子图边缘至少 5°）；`lenses` 的 `plan` 记录该组规划。原图先在 CPU 按面积缩小到子图中心像素密度的 2 倍，再在 GPU 核函数中双线性重采样到各子图；卷帘快门交付下按每幅照片的读出转角 `ω` 对每个子图像素做 3 步不动点去卷帘反向映射，直接抽取对齐参考姿态 `s = 0` 的透视子图消除转身切线错位。MoGe 逐张推理，按已知焦距恢复 z 偏移并换算成沿视线距离与子图法向，融合时将法向旋回像素读出线相机系（`normal_estimator` 记为 `las_knn_pca_surfel_moge2_vitl_readout_line`，静态快门为 `las_knn_pca_surfel_moge2_vitl`），法向朝向相机。

反向清洗 LAS 深度（`CleanLasAndEncodeNormals`）与法向门控/混合输出按以下流程执行：
1. **MoGe 边缘反向 Mask 与平面法向冲突门控**：在每个子图上检查 3×3 邻域对数深度跳变超过 `0.05` 或法向夹角超过 `35°` 的轮廓像素，投影回监督图格点并以 `5×5` 椭圆结构元（2 像素半径）膨胀成边缘屏蔽带，丢弃落入该边缘带内的 LAS 投影像素（计为 `moge_edge_dropped_pixels`），彻底规避前景边界穿透与轻微外参偏移造成的错位深度梯度；同时在非边缘带内若某像素同时存在置信度合格的 LAS 平面面元法向与有效融合 MoGe 法向且两者夹角超过 `35°`，将该像素法向清零为 `(0,0,0)`（计为 `normal_disagree_dropped_pixels`）。
2. **多面协同的确定性逆深度 RANSAC 清洗与实体平面救援**：在静态域且避开上述边缘带的像素上按 2 像素步长采样 `(1 / r_moge, 1 / r_las)`，对每张 90° 子图独立做 256 次固定随机序列 RANSAC 线性拟合（`1 / r_fit = s / r_moge + t`，至少 64 样本、32 内点且内点占比 ≥ 40%；以 `3.5 × MAD` 截断在 `[0.25, 0.45]` 的自适应对数阈值 `tau` 精炼）。检查覆盖像素的全部已接受透视面：若任一已接受面判定为内点（`|log(r_las / r_fit)| <= tau`）则保留原始 LAS 深度；否则取 `|log(r_las / r_fit)|` 最小的最佳支持面判定近景飞点（`log(r_las / r_fit) < -tau`，计为 `ransac_dropped_near_pixels`）或远距候选——若同时具有 LAS 平面法向与有效 MoGe 法向、两者夹角 `<= 25°` 且 `log(r_las / r_fit) <= 1.1`（约 `3×` 深度比），判定为单目尺度低估而非玻璃穿透并保留原始 LAS 深度（计为 `far_normal_rescued_pixels`），其余远距外点按穿透滤除（计为 `ransac_dropped_far_pixels`）；若无已接受子图覆盖，则保留非边缘带的已过滤 LAS 深度。
3. **高置信度平面 LAS 法向优先与 MoGe-2 回填**：在未触发上述 `> 35°` 非边缘法向冲突门控的静态允许像素上，若 LAS 深度经上述步骤保留且具备有效 LAS 平面面元法向（满足局部 PCA 平面性与 `MinIncidence >= 0.3`），直接采用精确且多视角绝对一致的 LAS 面元法向（计为 `las_normal_used_pixels`）；其余 LAS 空洞、MoGe 边缘带及非平面细节像素则由满足入射角要求的融合 MoGe 法向回填（计为 `moge_normal_fill_pixels`），但对未获上述实体平面救援且 `log(r_las / r_fit) > tau` 与 `r_las - r_fit > 0.25 m` 的玻璃/镜面穿透像素同步清零为 `(0,0,0)`（计为 `penetration_normal_dropped_pixels`）。

以上法向有效不强制要求对应 LAS 深度有效（如纯 MoGe 回填区）。不再执行 `12×12` 尺度场拟合或跨视角自由空间冲突删点。`supervision.json` 的 `source` 为 `las_cleaned_moge2_normals`（并写明 `depth_source = "las_moge_cleaned"`、`normal_source = "las_planar_moge2_fill"`），另含 `geometry_model`（模型包、引擎 SHA256 与文件身份、来源、子图尺寸、token 数）、各镜头子图规划及 `monocular` 统计（`las_input_pixels`、`moge_edge_dropped_pixels`、`ransac_dropped_near_pixels`、`ransac_dropped_far_pixels`、`far_normal_rescued_pixels`、`penetration_normal_dropped_pixels`、`normal_disagree_dropped_pixels`、`las_normal_used_pixels`、`moge_normal_fill_pixels`、`inference`）与逐图分面 RANSAC 记录。监督请求另含 `source: "moge"`、`pipeline: "las_cleaned_moge2_normals"`、`rolling_shutter` 与几何引擎身份。

manifest 的 request.supervision 记录 LAS、静态及额外排除掩码身份和导出参数。supervision 字段与 `supervision.json` 相同，保存按已注册照片原顺序排列的尺寸、文件字节数及有效像素数量。`check` 逐图回读，核对类型、尺寸、身份、有效数量、法向单位长度及其深度支撑（MoGe 单目监督的法向不要求深度）。关闭监督时不生成这些字段和文件。

manifest 保存请求、已注册图像身份、文件大小、初始化参数；启用静态掩码时另含 `static_masks`（保存 `cache/static_masks.json` 中除逐连通块明细 `component_rows` 之外的统计与规则摘要，包括 `sky` 字段）。卷帘校正被采用时 request 另含 `rolling_shutter`：`cache/shutter.json` 的文件身份与 `photos: "original"`。图像条目的 image_index 是原输入的 0 起始索引。quality 保存所选模型拟合验收和限值。省略 PLY 时不生成 initialization.json，清单中的 initialization 为 null。

manifest 和 quality 的 registration 保存 complete、selected_images、registered_images，以及 unregistered_images 名单（image_index、name、0 起始 camera）。默认首次停滞就处理已注册部分；可配置 `sfm_repair_rounds=1` 或 `2`，在补匹配轮数用完后处理已注册部分。子集收尾还会排除末尾标定可靠支撑不足的曝光，其原始索引和支撑记录保存在 sfm.calibration_selection。只有通过支撑检查的已注册照片参与后续优化、几何验收及配准；原有验收门限不变。quality.images 保留全部输入的顺序与 registered 标记，未注册条目的观测数为 0、误差统计为空。quality.passed 表示已重建部分通过验收，不表示全部照片注册成功；check 同时返回 registration。

quality.calibration 保存两镜头的初始与最终内参、主点偏移，以及每 20° 入射角区间的重投影误差。角度按初始标定计算，便于比较镜头中心与边缘；这些统计是拟合诊断，不代表独立精度。

quality.track_lengths 保存有效三维点的轨迹长度数量、中位数和 P90，tracks_at_least_3、tracks_at_least_8 保存至少 3、8 次观测的点数，便于结合每图观测数及 occupied_cells 比较配对策略的覆盖。

`result.json` 的 joint 保存联合优化候选的完整质量、点位移统计、逐图退化明细、各项接受条件及 failed_checks；候选被拒绝时保留所选原模型，验收阈值不因诊断而放宽。

默认在 LAS 对齐后执行卷帘快门校正和独立稀疏点清理，再导出最终所选模型。`result.json` 的 alignment、joint、sfm 是各阶段原报告；shutter 保存读出时间、读出转动与平移统计、各镜头读出一致性、留出误差、同组 LAS 平面误差、观测换算位移、各项检查及 used；cleanup 单独保存稀疏点清理与验收。配置 `"rolling_shutter": false` 时 shutter.enabled=false。quality 与 navigation_error_m 更新为最终模型统计。

清理候选未通过时保留清理前的模型，未采用原因明确记录。允许清理最多 1% 的已注册照片，且每镜头至少保留 98%、可靠 LAS 支撑轨迹至少保留 90%；最终照片名单以交付 registration 为准。清理不改变相机内参和位姿，也不改变 LAS 初始化 PLY。

同级 `.pending` 中完成写入及回读后重命名发布。同一请求的完整包可复用，不同请求拒绝覆盖。`check --output` 只依赖交付包，检查结构、引用、尺寸、PLY 和几何拟合，不代表独立精度或渲染效果。

`manifest.json` 的 `project` 记录选中的 PCC `.pro` 文件名。`check --output DIR --project FILE.pro`（可给路径，按文件名比较）同时核对交付包是否来自指定解算项目。

格式依据：[COLMAP output format](https://colmap.github.io/format.html)。

## 卷帘快门交付

卷帘校正被采用的模型交付原图与原掩码（默认按有效域无损裁剪），特征坐标为交付图像像素，位姿为参考位姿；`rolling_shutter.json` 给出每幅照片的读出运动。按全局快门读取时视图间仍有卷帘偏差，下游须使用该文件（定制 LFS 在 `--gut` 且未开启 undistort 时按卷帘相机训练，否则告警并按全局快门处理）。照片不做重采样：读出平移的像素偏移随深度变化，按 LAS 距离重采样会在近处动态物体和 LAS 空洞处错位。points3D.bin 的 error、quality.json 与 `check` 均按卷帘投影计算；深度和法向监督按每个像素自身读出线的相机投影 LAS；面元法向随位姿及面元中心所在读出线的转动变换到该读出相机（与覆盖像素的读出线相差不超过足迹的几个像素）。supervision.json 的 projection 为 `rolling_shutter_readout_line`，normal_estimator 为 `las_knn_pca_surfel_readout_line`。

`rolling_shutter.json` 是卷帘交付的相机接口：

```text
readout_axis   "rows" 或 "columns"，两镜头共用
cameras        [{camera_id, readout_s}]，readout_s 为整帧读出时间（秒），符号表示读出先后
images         [{image_id, camera_id, name, rotation_rad[3], velocity_m_s[3], translation_m[3]}]，与 images.bin 的 ID 一致；读取时 camera_id 可省略，存在即须与 images.bin 相符
```

读出比例 `s = (c - N/2)/N`，c 为像素沿读出轴的坐标（左上像素中心为 0.5），N 为该轴的图像尺寸。参考位姿为 images.bin 的 world-to-camera `R, t`（s = 0）。三维点 X 在读出比例 s 处的相机坐标为 `p(s) = (I + s[rotation_rad]×)(R X + t - s·translation_m)`，`translation_m = readout_s · velocity_m_s`；rotation_rad、velocity_m_s、translation_m 均在 OpenCV 参考相机坐标中，rotation_rad 与 translation_m 对应整帧读出。X 的像素 x 满足 `x = fisheye(p(s(x)))`，从 s = 0 起不动点迭代 3–4 次即收敛。该一阶模型在读出期间转动不超过约 3° 时足够精确。

原图交付的法向在世界坐标的 LAS 点云上估计，投影时变换到面元中心读出线的相机。

## 有效域裁剪交付

`run` 与 `export` 默认把每个镜头的照片、掩码与模型裁到该镜头有效域的外接窗口；`--no-crop-valid-domain` 交付整幅原图。裁剪使下游少处理鱼眼圆外恒为 0 的像素。窗口取该镜头全部已注册照片静态掩码非零像素的外接框，四周各留 16 px（不超出原图）；左上角对齐照片 JPEG 的 MCU 网格，照片按 DCT 系数无损裁剪，不重新编码，EXIF 等标记原样保留（EXIF 中记录的像素尺寸不改写）。掩码按窗口精确裁剪。窗口沿读出轴以图像中心对称，读出比例只按 `N'/N` 缩放（N、N' 为裁剪前后沿读出轴的尺寸）：`rolling_shutter.json` 的 `rotation_rad`、`translation_m` 与 `readout_s` 同乘 `N'/N`，`velocity_m_s` 与参考位姿不变，交付在裁剪后图像上按同一公式 `s = (c - N/2)/N` 使用。

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
