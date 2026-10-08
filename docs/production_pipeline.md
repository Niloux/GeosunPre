# 生产运行与验收

## 配置

`--config` 接受严格 TOML，未知字段直接报错。冻结配置版本为 6，全部字段如下。

| 字段 | 默认 | 语义 |
| --- | --- | --- |
| `expected_images` | 省略 | 选中 L/R 图像总数必须相等 |
| `calibration_frames` | 省略 | 包含首尾的标定帧号范围 |
| `calibration_frame_count` | 25 | 未指定范围时的自动窗口大小 |
| `workers` | 4 | CPU 前端处理与配准查询/联合求解线程数，1–16 |
| `registration_las` | 省略 | 绝对路径或相对输入目录的 LAS 路径；默认唯一 `ResultOut/*.las` |
| `quality_max_median_px` | 2.0 | 整体及各镜头拟合重投影 median 上限 |
| `quality_max_p90_px` | 4.0 | 整体及各镜头拟合重投影 P90 上限 |
| `registration_min_support_fraction` | 0.25 | 拟合区域的可靠表面支持比例下限 |
| `registration_max_median_m` | 0.1 | 支持区域点到面 median 上限 |
| `registration_max_p90_m` | 0.3 | 支持区域点到面 P90 上限 |

标定片段必须含两个镜头、至少两个不同帧号；没有显式范围时自动跳过只含单镜头的前缀。局部和全局求解使用同一套特征编号。验收条件冻结后不可在原工作目录放宽；改变配置使用新目录。

## 固定阶段

| 阶段 | 主要结果 |
| --- | --- |
| `prepare` | 输入清单、图像及导航身份、图像数量与标定窗口检查 |
| `masks` | 语义/设备掩码和带掩码的场景清单 |
| `features` | LoMa 特征缓存 |
| `matches` | 固定候选对与匹配缓存 |
| `observations` | 所有有效观测来源及内容哈希 |
| `calibration` | 四步局部受限标定与最终身份 |
| `sfm` | 全图像增量建图、三步全局 BA 与轨迹筛选 |
| `finalize` | 曝光完整且通过质量检查的视觉模型 |
| `register` | 导航/LAS 输入协议、全局配准、联合候选、验收后正式模型 |
| `export` | 可移动 COLMAP 数据集及回读验包结果，默认附带掩码、5 cm LAS 初始化点云 `sparse/0/points3D.ply` 和 `initialization.json` |

`--stage` 只执行所指定阶段，上游必须已完成。新任务从 `prepare` 或默认 `all` 开始。通常直接重复完整运行命令，完整缓存会被验证和复用。

## 验收与结果查看

视觉交付要求全部入选曝光恰好注册一次，原生 L/R 鱼眼模型有效，观测具有正深度、镜头在 160° 内单调，拟合误差和覆盖满足配方及场景阈值，内参没有触及优化边界。

全局配准要求足够的拟合点和诊断点、可靠表面支持、空间覆盖和点到面误差。联合更新同时检查整体、镜头、图像、FOV 区域和空间瓦片的拟合投影变化，以及固定点身份上的几何变化；内参、图像成员和观测拓扑必须保持不变。联合失败采用已验收全局模型，全局失败停止导出。程序异常或输入身份变化会报错，不被当作数值退化吞掉。

```text
WORK/
  pipeline_config.json
  pipeline_task.json
  frontend_weights.json      # 首次使用 LoMa 后生成
  segmentation_weights.json  # 首次使用 SegFormer 后生成
  pipeline_status.json
  run_result.json
  manifest.json
  calibration/
  sfm/final/
  registration/
    inputs.json
    protocol.json
    global/
    joint/
    final/
      sparse/0/
      inspection.html
      result.json
      quality_summary.json
      completion.json
  colmap/
```

`registration/final/inspection.html` 展示选择结果、质量概要和拒绝联合更新的理由，链接详细 JSON。`quality.json` 中的重投影统计使用拟合数据；LAS 空间诊断区域不用于选择最终结果。内容验包验证交付一致性，不能替代独立几何精度或 3DGS 渲染检查。

## 恢复与交付

工作目录首次冻结输入、配置、执行源码、依赖版本、设备配方、模型缓存和默认导出参数。权重在实际使用对应模型前冻结内容身份；只做准备时可以随后补齐缓存。一旦使用，改变权重必须新建目录。后续调用不带输入、配置和交付参数时继承原任务；重新指定参数必须一致。

完成的数值阶段在临时目录写完后原子发布。中断的未提交模型归档为 `*.interrupted-*`，续跑从上一个完整阶段开始；已提交结果的身份或内容变化会被拒绝。强制终止可能留下临时目录，不会把它当成已完成模型。

源码、依赖、配置、输入、权重和算法配方改变时重新开始；不存在旧配置迁移、旧产物自动收编或源码身份豁免。版本升级后请使用新工作目录。

```bash
# 一键继续默认任务
./run.sh

# 恢复到同一任务，保留交付选项
.venv/bin/geosunpre run --work work/场景

# 重新导出；缓存阶段不需要 GPU
.venv/bin/geosunpre export --work work/场景

# 换输出路径和交付方式，不修改任务默认交付
.venv/bin/geosunpre export --work work/场景 --output /path/to/other --image-mode link

# 检查冻结任务的默认交付路径，包含自定义输出路径
.venv/bin/geosunpre check --work work/场景
```

同一输出目录不能改变复制/链接、掩码、PLY 或体素大小选项；换一个 `--output` 即可。`check --output` 独立读取交付包，复制模式无需原始输入和工作目录。失败阶段及具体原因查看 `pipeline_status.json`，最近成功调用查看 `run_result.json`。
