# COLMAP 与 PLY 交付

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
```

COLMAP 小端二进制，`OPENCV_FISHEYE` 模型编号 5、8 个参数。相机 ID 1、2；每幅图像独立 frame，使用镜头对应的一传感器 rig。内部索引从 0 开始；仅导出已注册照片，按原输入顺序将图像/frame ID 连续重编号为 1 到注册数量，轨迹中的图像引用同步转换，内部缓存编号保持不变。

位姿 world-to-camera，COLMAP 四元数顺序 `w x y z`。图像名相对于 images，保留原图、尺寸、畸变与全部特征编号。点轨迹与图像引用相互一致，无三维点的特征用 uint64 最大值。下游须支持原生鱼眼模型。

points3D.bin 的 error 为最终所选模型在该点全部观测上的平均重投影距离，单位像素；联合优化后重新计算，交付回读时验证。

相机和稀疏点已在输入 LAS 坐标，下游不再应用配准变换。BIN 是视觉稀疏点及轨迹；PLY 是完整有效 LAS 的体素均值，点身份相互独立。默认体素 0.05 m，二进制小端，每点 float32 XYZ、零法向量 XYZ、uint8 RGB，共 27 字节。LAS 头部范围的坐标绝对值须不超过 16777 m，使 float32 舍入不超过 1 mm，否则明确报错。下游需明确选择此 PLY 初始化 3DGS。

掩码为原图尺寸单通道 PNG：静态 255、天空 128、人物及有效域外 0。SfM 只提取 255 区域；有效域按初始标定冻结。五视图融合无覆盖的格子对应原图像素也为 0，独立精修复制旧掩码时同步屏蔽这些像素。

LAS 深度和法向监督默认在导出阶段生成，`run`、独立 `export`、`refine`、`cleanup` 均可用 `--no-las-supervision` 关闭。监督使用最终所选相机和原始有效 LAS，以 50 万记录分块、GPU 有界批次逐点投影，几何计算为双精度；不读取初始化 PLY。监督图默认最长边 1600，不放大原图，相机内参按输出尺寸缩放。深度是沿视线距离，uint16 毫米，0 表示无效，0.05 m 以下及 65.535 m 以上留空。法向是 OpenCV 相机坐标下朝向相机的单位向量，RGB 为 `round(127.5*(n+1))`，黑色无效。Spirula 需显式设置 `input_depth_is_ray_depth=true`；图像仍保留原始鱼眼畸变。

每次监督导出只扫描、过滤一次原始 LAS。原始整数 XYZ 的最大占用能放入预算时，坐标一次上传后在 GPU 复用；否则写入交付临时目录的紧凑坐标文件，按相机批次双缓冲读取、上传与投影，成功后删除该文件。`--supervision-cache-mb` 设置 GPU 坐标缓存上限（默认 1024 MiB，1–4096），实际预算还限制为开始时空闲显存的四分之一；深度图等工作缓冲单独分配。使用文件缓存时，主机和 GPU 传输缓冲各固定约 12 MB，内存不随 LAS 点数增长。`supervision.json` 的 `point_cache` 记录存储方式、扫描次数、坐标字节数、上传量和准备耗时。缓存预算仅控制执行资源，不改变监督请求身份或像素含义。

先剔除近处表面空隙中漏出的远点，再以充分且深度一致的邻居补小孔。深度断层、语义边界默认留出 6 个监督图像素；局部三维面残差不超过 1 cm、法向与视线夹角余弦绝对值至少 0.3 时保留深度。法向另需两种邻域尺度相差不超过 10°、粗尺度面残差不超过 1.5 cm。缺测允许单图完全为空，但整包必须有有效深度和法向。以上是几何一致性筛选，不证明真实可见性或自动识别玻璃；`--supervision-mask-dir` 可按原图身份追加仅影响几何的排除掩码，原 RGB 掩码不因 LAS 缺测而改变。

manifest 的 request.supervision 记录算法版本、LAS、静态及额外排除掩码身份和导出参数；supervision 字段与 `supervision.json` 相同，保存按已注册照片原顺序排列的尺寸、文件字节数及有效像素数量。`check` 逐图回读，核对类型、尺寸、身份、有效数量、法向单位长度及其深度支撑。关闭监督时不生成这些字段和文件，旧包仍可回读。

manifest 协议版本 1，保存请求、已注册图像身份、文件大小、初始化参数。图像条目的 image_index 是原输入的 0 起始索引。quality 保存所选模型拟合验收和限值。省略 PLY 时不生成 initialization.json，清单中的 initialization 为 null。

manifest 和 quality 的 registration 保存 complete、selected_images、registered_images，以及 unregistered_images 名单（image_index、name、0 起始 camera）。默认首次停滞就处理已注册部分；可配置 `sfm_repair_rounds=1` 或 `2`，在补匹配轮数用完后处理已注册部分。子集收尾还会排除末尾标定可靠支撑不足的曝光，其原始索引和支撑记录保存在 sfm.calibration_selection。只有通过支撑检查的已注册照片参与后续优化、几何验收及配准；原有验收门限不变。quality.images 保留全部输入的顺序与 registered 标记，未注册条目的观测数为 0、误差统计为空。quality.passed 表示已重建部分通过验收，不表示全部照片注册成功；check 同时返回 registration。旧版完整包可以没有 registration。

quality.calibration 保存两镜头的初始与最终内参、主点偏移，以及每 20° 入射角区间的重投影误差。角度按初始标定计算，便于比较镜头中心与边缘；这些统计是拟合诊断，不代表独立精度。

quality.track_lengths 保存有效三维点的轨迹长度数量、中位数和 P90，tracks_at_least_3、tracks_at_least_8 保存至少 3、8 次观测的点数，便于结合每图观测数及 occupied_cells 比较配对策略的覆盖。

工作目录 `result.json` 的 joint 保存联合优化候选的完整质量、点位移统计、逐图退化明细、各项接受条件及 failed_checks；候选被拒绝时保留所选原模型，验收阈值不因诊断而放宽。

默认在 LAS 对齐后独立清理稀疏点，再导出最终所选模型；原图精修默认关闭，`--refine` 或配置 `"refine": true` 在清理前启用。`result.json` 的 alignment、joint、sfm 是各阶段原报告；refinement 保存精修的启用状态、留出误差、同组 LAS 平面误差及相机优化验收，cleanup 单独保存稀疏点清理与验收。关闭精修时 refinement.enabled=false，且不生成精修缓存。quality 与 navigation_error_m 更新为最终模型统计。

精修未启用或相机候选未通过时，对对齐模型执行清理；清理候选未通过时保留清理前的模型，未采用原因明确记录。允许清理最多 1% 的已注册照片，且每镜头至少保留 98%、可靠 LAS 支撑轨迹至少保留 90%；最终照片名单以交付 registration 为准。清理不改变相机内参和位姿，也不改变 LAS 初始化 PLY。独立 `refine` 只执行相机精修，独立 `cleanup` 只执行清理，两者均写入新工作目录。

同级 `.pending` 中完成写入及回读后重命名发布。同一请求的完整包可复用，不同请求拒绝覆盖。`check --output` 只依赖交付包，检查结构、引用、尺寸、PLY 和几何拟合，不代表独立精度或渲染效果。

格式依据：[COLMAP output format](https://colmap.github.io/format.html)。
