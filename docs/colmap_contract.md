# COLMAP 交付契约

正式输出为原图、原生鱼眼模型、质量与来源记录，默认复制图像并附带有效域掩码和 LAS 降采样后的 3DGS 初始化 PLY。使用 `--no-include-ply` 可省略初始化点云。

```text
colmap/
  images/L/*.JPG
  images/R/*.JPG
  sparse/0/
    cameras.bin
    images.bin
    points3D.bin             # SfM 稀疏点与观测轨迹
    rigs.bin
    frames.bin
    points3D.ply             # LAS 体素降采样，用于 3DGS 初始化，默认启用
  masks/L/*.JPG.png          # 默认启用
  masks/R/*.JPG.png
  quality.json
  initialization.json       # LAS 来源、降采样参数、颜色、点数与文件大小
  colmap_export.json
```

## 图像、几何与坐标

L/R 各共享一组 `OPENCV_FISHEYE` 内参，每次曝光使用独立 frame。模型保留原图尺寸、像素坐标、观测轨迹和所有稀疏点，五份 COLMAP 二进制逐字节复制；不去畸变、不缩放。下游训练器须支持该鱼眼模型，并按需读取掩码。

图像名相对于 `images/`，位姿使用 COLMAP 的 world-to-camera 约定。相机和三维点已经变换到输入 LAS 的 XYZ 坐标；下游不再应用工作目录中的 Sim3。LAS 不修改。输出的坐标单位按米解释，绝对 CRS 和绝对地理精度尚需外部核对。

`points3D.ply` 是完整 LAS 的体素降采样结果，默认体素边长 `0.05 m`，可通过 `--ply-voxel-size` 调整。按块扫描、在磁盘分桶归并，每个体素保留坐标与 RGB 均值；排除 LAS withheld 和 class-7 噪声记录。支持 8-bit 值存于 uint16 和标准 16-bit RGB；无 RGB 的 LAS 使用灰色 127。输出字段为 XYZ、零法向量和 RGB，符合常见 3DGS PLY 点云读取格式，参见 [官方读取器](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/dataset_readers.py)。

初始化点与相机共享输入 LAS 的坐标框架。`points3D.bin` 继续保存 SfM 稀疏点与观测轨迹；它与初始化 PLY 的点数和编号独立。下游应明确使用 `points3D.ply` 初始化高斯；优先读取 BIN 的训练器需要显式指定 PLY。

掩码为单通道 8-bit PNG：静态区域为 255（白色），天空为 128（灰色），动态障碍物（当前分割为人物）为 0（黑色）。SfM 特征提取只接受值为 255 的像素。掩码结合初始标定下的 160° FOV；FOV 不随 BA 改变，FOV 外统一为 0。工作目录的 `sfm/masks/valid/` 保存语义掩码，导出时保留天空的 128 并叠加 FOV。

## 身份、质量与发布

`colmap_export.json` 使用协议 4，保存结构化导出请求、文件大小、图像尺寸、初始化参数及来源。请求直接比较源文件元数据和参数，来源说明 LAS 路径、场景配置以及实际选择的 `pose_las` 或 `global_registration`。`quality.json` 保存所选模型的图像诊断与几何验收。

导出在目标同级临时目录完成，回读检查成员、引用、尺寸、有限参数及 PLY 点数后原子发布。同一请求验包通过后直接复用，不重复拷图和生成掩码。已存在不同请求或没有清单的非空输出拒绝覆盖。无旧协议兼容或三文件/文本模型自动转换。

`copy` 模式整体可移动；`link` 模式采用相对符号链接，依赖原图继续存在。

```bash
.venv/bin/geosunpre check --output /path/to/colmap
```

复制模式只需交付包；来源信息用于追踪，接收方无需持有原输入和工作目录。`check --work` 读取任务的默认输出路径。验包核对文件大小与模型结构，允许复制或传输后修改时间变化。这些检查不证明文件逐字节相等，也不代表独立几何精度或下游渲染效果。数据划分及训练参数由下游决定。
