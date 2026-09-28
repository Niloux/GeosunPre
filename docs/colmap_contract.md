# COLMAP 输出契约

正式交付内容为原图与 COLMAP 场景模型。下游训练参数、数据划分和初始化点筛选由使用方决定。

## 文件与语义

```text
colmap/
  images/                  # 内容一致的原图拷贝，或显式选择的链接
  sparse/0/
    cameras.bin            # 相机模型、尺寸、内参
    images.bin             # 每次曝光位姿、二维观测、三维点引用
    points3D.bin           # 全部源模型三维点及观测轨迹
    rigs.bin
    frames.bin
  colmap_export.json       # 本项目的附加内容身份
  quality.json             # 完整流程附带的评估；通用导出可省略
  masks/                   # --include-masks
  sparse/0/points3D.ply     # --include-ply
```

`images.bin` 中的名称相对于 `images/`，位姿为 COLMAP 的 world-to-camera 约定。导出保持源模型相机类型、像素坐标和图像字节，不去畸变、不缩放、不变换坐标系。独立曝光继续使用独立 frame。视觉重建尺度任意，不代表米制或导航坐标。

五份二进制齐全时逐字节复制；导入三文件或文本模型时由 PyCOLMAP 写出完整二进制模型。导出检查内部引用、图像成员和尺寸、有限参数及文件内容哈希。

可选 PLY 包含全部源模型三维点，不设最少点数，也不执行下游初始化过滤。

可选掩码以白色表示有效、黑色表示排除，组合场景清单的 FOV 与语义/设备掩码。FOV 使用特征提取时的初始标定，不随 BA 悄悄改变。工作目录的 `masks_v2/valid` 只含语义与设备排除；导出 `masks/` 才是组合后的完整有效域。掩码不是 COLMAP 稀疏模型的必需文件。

## 发布与独立校验

默认 `--image-mode copy`，整个包可移动到其他目录或机器；`--image-mode link` 使用相对符号链接，依赖原图路径继续存在。

```bash
geosunpre check --output /path/to/colmap
```

也可用 `--work` 检查 `WORK/colmap`。复制模式下只读取导出包，来源哈希保存在 `provenance` 中用于追踪，不要求接收方持有原始工作目录。

导出在目标同级临时目录中生成、回读并校验，成功后发布整个目录。失败不发布半成品；强制终止可能留下隐藏临时目录，续跑不复用它。已有目录只在身份和选项相同时复用；不同内容或无导出身份的非空目录拒绝覆盖。

`colmap_export.json` 使用版本 1，COLMAP 本身无需解析它。旧版流程已经交付的同版本数据集仍可独立验证。内部一致性与内容校验不代替几何质量验收。
