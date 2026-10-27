# COLMAP 与 PLY 交付

交付直接位于 `--output` 根目录，下游（LFS、Spirula）的数据路径即为该目录。

```text
OUTPUT/
  images/{L,R}/**.JPG
  masks/{L,R}/**.JPG.png
  depths/{L,R}/**.JPG.png       # 默认生成，--no-las-supervision 省略
  normals/{L,R}/**.JPG.png      # 默认生成，--no-las-supervision 省略
  sparse/0/{cameras,images,points3D,rigs,frames}.bin
  sparse/0/points3D.ply
  manifest.json
  quality.json
  initialization.json
  supervision.json            # 默认生成，--no-las-supervision 省略
  rolling_shutter.json        # 卷帘校正被采用时生成
  result.json                 # 各阶段报告，与工作目录 result.json 相同，不参与 check
```

COLMAP 小端二进制，`OPENCV_FISHEYE` 模型编号 5、8 个参数。相机 ID 1、2；每幅图像独立 frame，使用镜头对应的一传感器 rig。内部索引从 0 开始；仅导出已注册照片，按原输入顺序将图像/frame ID 连续重编号为 1 到注册数量，轨迹中的图像引用同步转换，内部缓存编号保持不变。

位姿 world-to-camera，COLMAP 四元数顺序 `w x y z`。图像名相对于 images，保留原图名、鱼眼畸变与全部特征编号。照片和掩码不做重采样：默认无损裁到各镜头有效域窗口（见“有效域裁剪交付”），`--no-crop-valid-domain` 时为整幅输入照片拷贝；默认为 L/R 原图，配置 `"enhanced_photos": true` 且输入有 `Image/enhanced` 同名图时为该 enhanced 图。卷帘快门校正被采用时，位姿为读出中线时刻的参考位姿，并附 `rolling_shutter.json`，见下节。点轨迹与图像引用相互一致，无三维点的特征用 uint64 最大值。下游须支持原生鱼眼模型。

points3D.bin 的 error 为最终所选模型在该点全部观测上的平均重投影距离，单位像素；联合优化后重新计算，交付回读时验证。

相机和稀疏点已在输入 LAS 坐标，下游不再应用配准变换。BIN 是视觉稀疏点及轨迹；PLY 是完整有效 LAS 的体素均值，点身份相互独立。默认体素 0.05 m，二进制小端，每点 float32 XYZ、零法向量 XYZ、uint8 RGB，共 27 字节。LAS 头部范围的坐标绝对值须不超过 16777 m，使 float32 舍入不超过 1 mm，否则明确报错。下游需明确选择此 PLY 初始化 3DGS。

掩码为与交付照片同尺寸的单通道 PNG：静态 255、天空 128、人物及有效域外 0。SfM 只提取 255 区域；有效域按初始标定冻结。五视图融合无覆盖的格子对应原图像素也为 0，独立精修复制旧掩码时同步屏蔽这些像素。

LAS 深度和法向监督默认在导出阶段生成，`run`、独立 `export`、`refine`、`cleanup` 均可用 `--no-las-supervision` 关闭。监督使用最终所选相机和原始有效 LAS：先在点云上为每点估计法向与面元足迹，再以 50 万记录分块、GPU 有界批次投影面元，位姿与投影为双精度；不读取初始化 PLY。监督图默认最长边 1600，不放大原图，相机内参按输出尺寸缩放。深度是沿视线距离，uint16 毫米，0 表示无效，0.05 m 以下及 65.535 m 以上留空。法向是 OpenCV 相机坐标下朝向相机的单位向量，RGB 为 `round(127.5*(n+1))`，黑色无效。Spirula 需显式设置 `input_depth_is_ray_depth=true`；图像仍保留原始鱼眼畸变。

每次监督导出只扫描、过滤一次原始 LAS，保留的整数 XYZ 写入交付临时目录的紧凑坐标文件，同时沿 LAS 头部范围最长轴统计 65536 格直方图。法向估计按该轴分成若干片，每片连同两侧 15 cm 的邻域点不超过开始时空闲显存一半可容纳的点数（至多 6400 万），单片时整片一次处理；每片结果为每点 20 字节的面元（整数 XYZ、八面体编码的世界法向、足迹半径），之后删除坐标文件。面元总量能放入预算时一次上传后在 GPU 复用；否则写入紧凑面元文件，按相机批次双缓冲读取、上传与投影，成功后删除。`--supervision-cache-mb` 设置 GPU 面元缓存上限（默认 2048 MiB，1–4096），实际预算还限制为空闲显存的四分之一；深度图等工作缓冲单独分配。使用文件缓存时，主机和 GPU 传输缓冲各固定约 20 MB；法向分片的主机缓冲以片容量为界，内存不随 LAS 点数无界增长。静态掩码读取缩放、监督图 PNG 写出及交付回读与 GPU 并行，最多各 6 幅在途。`supervision.json` 的 `point_cache` 记录存储方式、扫描次数、面元字节数、上传量和准备耗时，`surfels` 记录法向估计参数、分片数、有法向点数、足迹与平面均方根的中位数及耗时。缓存预算仅控制执行资源，不改变监督请求身份或像素含义。

点云法向：以 3 cm 哈希网格在 6 cm 范围内求每点的 16 个最近邻，第 16 近邻距离为点间距（不足 16 个时取 6 cm）；不足 10 个近邻的点没有法向。法向取以该点为球心、半径为点间距与 3 cm 中较大者（不超过 6 cm）的球内点做 PCA：第一次全部点，第二次只保留距平面 max(2 cm, 2.5σ) 内的点，第三次收敛到 2 cm 内点带。有效平面须至少 10 个内点、内点占球内点至少 60%、法向方向的均方根 σ 不超过 1 cm、次特征值不小于最大特征值的 5%（排除线状支撑），且该点距平面不超过 2 cm；点间距还不得超过其近邻点间距中位数的 2 倍，表面前方的零散回波因近邻都在远处的密集表面上而被排除。有法向且 16 个近邻都在范围内的点得到足迹圆盘，半径为 2 × min(点间距, 近邻点间距中位数) / 4，至多 5 cm；其他点只投影到自身像素，孤立回波仍在投影后被剔除。

投影分两遍：第一遍求每个像素最近的视线距离；面元覆盖视线与其切平面交点落在圆盘内的像素（离中心不超过 8 个监督图像素，交点视线与法向夹角余弦须大于 0.05），点自身所在像素总被覆盖。第二遍把距最近距离 1 cm 加 1% 以内的全部面元按 `1-(偏离/半径)²`（下限 0.05）加权平均，得到该像素的深度和法向；法向在相机坐标中朝向相机，有法向面元的权重须占一半以上，且单位法向的加权平均长度不小于 0.95，否则该像素只有深度。像素坐标、视线及卷帘读出线与此前相同。

投影后先剔除缺少局部深度支撑的孤立点（规则同下述深度有效性），再剔除近处表面空隙中漏出的远点，最后以充分且深度一致的邻居补小孔。孤立异常点因此既不挖空周围表面，也不触发断层。遮挡断层取两侧均有支撑的台阶：5×5 邻域内背景比前景深出前景深度的 10% 加 5 cm；斜看的地面、天花板和侧墙相邻像素间的深度渐变不构成断层。遮挡断层两侧的余量不对称：背景侧只去掉台阶处前景 2 像素内的背景像素；前景侧去掉位于这些台阶像素前方、距离不超过 `ceil(1 cm × fx / 深度)` 像素（至多 24）的前景像素，fx 为监督图焦距。LAS 前景轮廓因横向测距噪声和多次扫描叠加，比照片中的轮廓向外扩出约 1 cm，前景侧余量覆盖这部分；在室内场景实测，4 m 内照片边缘与保留的 LAS 前景边缘相差 0–1.3 像素，整体平移小于 1 像素。语义掩码边界（静态掩码中的人物、天空及有效域边界）默认留出 6 个监督图像素，由 `--supervision-boundary` 设置。深度和法向分别判断有效性：深度在 5×5 邻域中须有至少 4 个深度一致的像素（含自身，距离差不超过中心深度的 2% 加 1 cm），不依赖法向是否可求。法向取上述混合面元法向，只在该像素保留投影所得深度（非补孔）时有效，且法向与视线夹角余弦绝对值至少 0.3。法向有效时深度必须有效，黑色法向不表示对应深度无效。缺测允许单图完全为空，但整包必须有有效深度和法向。以上是几何一致性筛选，不证明真实可见性或自动识别玻璃；`--supervision-mask-dir` 可按原图身份追加仅影响几何的排除掩码，原 RGB 掩码不因 LAS 缺测而改变。

manifest 的 request.supervision 记录算法版本、LAS、静态及额外排除掩码身份和导出参数；当前监督算法版本为 4，版本 1–3 的旧交付仍可回读。supervision 字段与 `supervision.json` 相同，保存按已注册照片原顺序排列的尺寸、文件字节数及有效像素数量。`check` 逐图回读，核对类型、尺寸、身份、有效数量、法向单位长度及其深度支撑。关闭监督时不生成这些字段和文件，旧包仍可回读。

manifest 协议版本 1，保存请求、已注册图像身份、文件大小、初始化参数。卷帘校正被采用时 request 另含 `rolling_shutter`：`cache/shutter.json` 的文件身份与 `photos: "original"`。图像条目的 image_index 是原输入的 0 起始索引。quality 保存所选模型拟合验收和限值。省略 PLY 时不生成 initialization.json，清单中的 initialization 为 null。

manifest 和 quality 的 registration 保存 complete、selected_images、registered_images，以及 unregistered_images 名单（image_index、name、0 起始 camera）。默认首次停滞就处理已注册部分；可配置 `sfm_repair_rounds=1` 或 `2`，在补匹配轮数用完后处理已注册部分。子集收尾还会排除末尾标定可靠支撑不足的曝光，其原始索引和支撑记录保存在 sfm.calibration_selection。只有通过支撑检查的已注册照片参与后续优化、几何验收及配准；原有验收门限不变。quality.images 保留全部输入的顺序与 registered 标记，未注册条目的观测数为 0、误差统计为空。quality.passed 表示已重建部分通过验收，不表示全部照片注册成功；check 同时返回 registration。旧版完整包可以没有 registration。

quality.calibration 保存两镜头的初始与最终内参、主点偏移，以及每 20° 入射角区间的重投影误差。角度按初始标定计算，便于比较镜头中心与边缘；这些统计是拟合诊断，不代表独立精度。

quality.track_lengths 保存有效三维点的轨迹长度数量、中位数和 P90，tracks_at_least_3、tracks_at_least_8 保存至少 3、8 次观测的点数，便于结合每图观测数及 occupied_cells 比较配对策略的覆盖。

`result.json` 的 joint 保存联合优化候选的完整质量、点位移统计、逐图退化明细、各项接受条件及 failed_checks；候选被拒绝时保留所选原模型，验收阈值不因诊断而放宽。

默认在 LAS 对齐后执行卷帘快门校正和独立稀疏点清理，再导出最终所选模型；原图精修默认关闭，`--refine` 或配置 `"refine": true` 在卷帘校正前启用。`result.json` 的 alignment、joint、sfm 是各阶段原报告；refinement 保存精修的启用状态、留出误差、同组 LAS 平面误差及相机优化验收；shutter 保存读出时间、读出转动与平移统计、各镜头读出一致性、留出误差、同组 LAS 平面误差、观测换算位移、各项检查及 used；cleanup 单独保存稀疏点清理与验收。关闭精修时 refinement.enabled=false，且不生成精修缓存；配置 `"rolling_shutter": false` 时 shutter.enabled=false。quality 与 navigation_error_m 更新为最终模型统计。

精修未启用或相机候选未通过时，对对齐模型执行清理；清理候选未通过时保留清理前的模型，未采用原因明确记录。允许清理最多 1% 的已注册照片，且每镜头至少保留 98%、可靠 LAS 支撑轨迹至少保留 90%；最终照片名单以交付 registration 为准。清理不改变相机内参和位姿，也不改变 LAS 初始化 PLY。独立 `refine` 只执行相机精修，独立 `cleanup` 只执行清理，两者均写入新工作目录。

同级 `.pending` 中完成写入及回读后重命名发布。同一请求的完整包可复用，不同请求拒绝覆盖。`check --output` 只依赖交付包，检查结构、引用、尺寸、PLY 和几何拟合，不代表独立精度或渲染效果。

格式依据：[COLMAP output format](https://colmap.github.io/format.html)。

## 卷帘快门交付

卷帘校正被采用的模型交付原图与原掩码（默认按有效域无损裁剪），特征坐标为交付图像像素，位姿为参考位姿；`rolling_shutter.json` 给出每幅照片的读出运动。按全局快门读取时视图间仍有卷帘偏差，下游须使用该文件（spirula 的 `3dgut` 按卷帘相机训练）。照片不做重采样：读出平移的像素偏移随深度变化，按 LAS 距离重采样会在近处动态物体和 LAS 空洞处错位。旧版按参考位姿重采样照片的 `rectified` 交付不再生成，仍可 `check` 回读。points3D.bin 的 error、quality.json 与 `check` 均按卷帘投影计算；深度和法向监督按每个像素自身读出线的相机投影 LAS；面元法向随位姿及面元中心所在读出线的转动变换到该读出相机（与覆盖像素的读出线相差不超过足迹的几个像素）。supervision.json 的 projection 为 `rolling_shutter_readout_line`，normal_estimator 为 `las_knn_pca_surfel_readout_line`，监督请求另含 `readout_normals_version`。

`rolling_shutter.json`（version 1）是卷帘交付的相机接口：

```text
readout_axis   "rows" 或 "columns"，两镜头共用
cameras        [{camera_id, readout_s}]，readout_s 为整帧读出时间（秒），符号表示读出先后
images         [{image_id, camera_id, name, rotation_rad[3], velocity_m_s[3], translation_m[3]}]，与 images.bin 的 ID 一致；读取时 camera_id 可省略，存在即须与 images.bin 相符
```

读出比例 `s = (c - N/2)/N`，c 为像素沿读出轴的坐标（左上像素中心为 0.5），N 为该轴的图像尺寸。参考位姿为 images.bin 的 world-to-camera `R, t`（s = 0）。三维点 X 在读出比例 s 处的相机坐标为 `p(s) = (I + s[rotation_rad]×)(R X + t - s·translation_m)`，`translation_m = readout_s · velocity_m_s`；rotation_rad、velocity_m_s、translation_m 均在 OpenCV 参考相机坐标中，rotation_rad 与 translation_m 对应整帧读出。X 的像素 x 满足 `x = fisheye(p(s(x)))`，从 s = 0 起不动点迭代 3–4 次即收敛。该一阶模型在读出期间转动不超过约 3° 时足够精确。

原图交付的法向在世界坐标的 LAS 点云上估计，投影时变换到面元中心读出线的相机。监督请求的 `readout_normals_version=2` 区分此几何约定（版本 1 为按参考相机拟合的图像邻域法向）；旧交付仍可回读，新导出使用新目录。

## 有效域裁剪交付

`run`、`export`、`refine`、`cleanup` 默认把每个镜头的照片、掩码与模型裁到该镜头有效域的外接窗口；`--no-crop-valid-domain` 交付整幅原图，`--crop-valid-domain` 显式开启，二者冲突。裁剪使下游少处理鱼眼圆外恒为 0 的像素。窗口取该镜头全部已注册照片静态掩码非零像素的外接框，四周各留 16 px（不超出原图）；左上角对齐照片 JPEG 的 MCU 网格，照片按 DCT 系数无损裁剪，不重新编码，EXIF 等标记原样保留（EXIF 中记录的像素尺寸不改写）。掩码按窗口精确裁剪。窗口沿读出轴以图像中心对称，读出比例只按 `N'/N` 缩放（N、N' 为裁剪前后沿读出轴的尺寸）：`rolling_shutter.json` 的 `rotation_rad`、`translation_m` 与 `readout_s` 同乘 `N'/N`，`velocity_m_s` 与参考位姿不变，交付在裁剪后图像上按同一公式 `s = (c - N/2)/N` 使用。

`cameras.bin` 的宽高为窗口尺寸，主点减去窗口原点，畸变参数不变；`images.bin` 的特征坐标减去窗口原点，`points3D.bin` 的 error 按裁剪后的投影重新计算。深度、法向监督按裁剪后的相机生成，静态及额外排除掩码先裁到窗口再缩放到监督图；监督图像素只有在其原图面积内全部有效时才有效（对无效像素做面积缩放，结果须严格为 0）。`manifest.json` 的 request 另含 `crop`（`mode: "valid_domain"` 与两镜头窗口），监督请求另含 `crop_windows`，因此裁剪与未裁剪交付不会被视为同一请求。`check` 对裁剪交付按同样规则回读。
