# COLMAP 与 PLY 交付

```text
OUTPUT/
  images/{L,R}/**.JPG
  masks/{L,R}/**.JPG.png
  sparse/0/{cameras,images,points3D,rigs,frames}.bin
  sparse/0/points3D.ply
  manifest.json
  quality.json
  initialization.json
```

COLMAP 小端二进制，`OPENCV_FISHEYE` 模型编号 5、8 个参数。相机 ID 1、2；每幅图像独立 frame，使用镜头对应的一传感器 rig。内部索引从 0 开始；仅导出已注册照片，按原输入顺序将图像/frame ID 连续重编号为 1 到注册数量，轨迹中的图像引用同步转换，内部缓存编号保持不变。

位姿 world-to-camera，COLMAP 四元数顺序 `w x y z`。图像名相对于 images，保留原图、尺寸、畸变与全部特征编号。点轨迹与图像引用相互一致，无三维点的特征用 uint64 最大值。下游须支持原生鱼眼模型。

points3D.bin 的 error 为最终所选模型在该点全部观测上的平均重投影距离，单位像素；联合优化后重新计算，交付回读时验证。

相机和稀疏点已在输入 LAS 坐标，下游不再应用配准变换。BIN 是视觉稀疏点及轨迹；PLY 是完整有效 LAS 的体素均值，点身份相互独立。默认体素 0.05 m，二进制小端，每点 float32 XYZ、零法向量 XYZ、uint8 RGB，共 27 字节。下游需明确选择此 PLY 初始化 3DGS。

掩码为原图尺寸单通道 PNG：静态 255、天空 128、人物及有效域外 0。SfM 只提取 255 区域；有效域按初始标定冻结。

manifest 协议版本 1，保存请求、已注册图像身份、文件大小、初始化参数。图像条目的 image_index 是原输入的 0 起始索引。quality 保存所选模型拟合验收和限值。省略 PLY 时不生成 initialization.json，清单中的 initialization 为 null。

manifest 和 quality 的 registration 保存 complete、selected_images、registered_images，以及 unregistered_images 名单（image_index、name、0 起始 camera）。补匹配两轮后允许部分注册，只有已注册照片参与后续优化、几何验收及配准；原有验收门限不变。quality.images 保留全部输入的顺序与 registered 标记，未注册条目的观测数为 0、误差统计为空。quality.passed 表示已重建部分通过验收，不表示全部照片注册成功；check 同时返回 registration。旧版完整包可以没有 registration。

quality.calibration 保存两镜头的初始与最终内参、主点偏移，以及每 20° 入射角区间的重投影误差。角度按初始标定计算，便于比较镜头中心与边缘；这些统计是拟合诊断，不代表独立精度。

quality.track_lengths 保存有效三维点的轨迹长度数量、中位数和 P90，tracks_at_least_3、tracks_at_least_8 保存至少 3、8 次观测的点数，便于结合每图观测数及 occupied_cells 比较配对策略的覆盖。

工作目录 `result.json` 的 joint 保存联合优化候选的完整质量、点位移统计、逐图退化明细、各项接受条件及 failed_checks；候选被拒绝时保留所选原模型，验收阈值不因诊断而放宽。

同级 `.pending` 中完成写入及回读后重命名发布。同一请求的完整包可复用，不同请求拒绝覆盖。`check --output` 只依赖交付包，检查结构、引用、尺寸、PLY 和几何拟合，不代表独立精度或渲染效果。

格式依据：[COLMAP output format](https://colmap.github.io/format.html)。
