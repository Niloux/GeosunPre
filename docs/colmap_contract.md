# COLMAP 输出契约

正式交付边界是原始采集数据到 COLMAP 场景数据。下游训练参数、训练/测试划分、初始化点数要求属于使用方。

## 文件与语义

```text
colmap/
  images/                  # 原图链接或内容一致的拷贝
  sparse/0/
    cameras.bin            # 相机模型、尺寸、内参
    images.bin             # 每次曝光位姿、二维观测、三维点引用
    points3D.bin           # 全部源模型三维点及观测轨迹
    rigs.bin
    frames.bin
  colmap_export.json       # 本项目附加的内容身份
  quality.json             # 构建入口附带的阶段评估；通用导出可省略
  masks/                   # --include-masks
  sparse/0/points3D.ply    # --include-ply
```

`images.bin` 中的名称相对于 `images/`；位姿采用 COLMAP 的 world-to-camera 约定。保留源模型的相机模型与像素坐标，不进行隐式去畸变、缩放或坐标系变换。现有视觉重建为任意尺度，不代表米制或导航坐标。独立曝光继续使用独立 frame。

五份二进制文件齐全时逐字节复制；导入旧版三文件模型或文本模型时，由 PyCOLMAP 写出当前版本的完整模型。导出校验检查模型内部引用一致性、图像成员和尺寸、有限参数以及全部文件内容哈希。

默认交付完整稀疏模型。可选 PLY 包含同一模型的全部三维点，不应用历史 LFS 的距离/误差/轨迹长度筛选，不设 10,000 点下限。

可选掩码以白色表示有效像素、黑色表示排除像素，组合场景清单的 FOV 和语义掩码。其 FOV 依据特征提取时的场景标定，不会悄悄切换为 BA 后的新有效域。掩码是附加文件，不是 COLMAP 稀疏模型的必需部分。

## 交付模式与校验

`--image-mode link` 是默认的本机模式，需要保留原图；`--image-mode copy` 复制原图，整个包可移动到其他目录或机器。两种模式均保留原始图像字节。

```bash
geosunpre check-colmap --output /path/to/colmap
```

该命令不读取原始工作目录、特征缓存、源重建或 LFS 配置。来源哈希保存在 `provenance` 中供追踪；独立检查验证导出内容，不声称重新审计已经离线的上游文件。`check-computed-baseline` 是同一检查的兼容命令名；提供 `--work` 时默认检查 `WORK/colmap`。

导出先在目标目录同级的临时目录中生成、回读并校验，再发布完整目录。失败不会把半成品发布成最终数据集；强制终止进程可能留下以目标名称为前缀的隐藏临时目录，但续跑不复用它。已发布目录只有身份完全一致时才可复用，改变模型、图像、选项或来源应选择新的目录。旧 `baseline_export.json`、`computed_baseline.json` 目录不会原地迁移。

`colmap_export.json` 是版本 1 的本项目清单，COLMAP 本身不需要解析它。内容身份与内部一致性检查不代替几何质量验收；空评估查询显式报告 `no_queries`。
