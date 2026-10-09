# 生产运行

## 配置与阶段

`--config` 接受严格 TOML，未知字段报错。没有标定帧号或场景模式；[通用示例](../configs/default.toml)可省略，默认值相同。

| 字段 | 默认 | 含义 |
| --- | --- | --- |
| `workers` | 4 | LAS 查询与联合求解线程数，1–16 |
| `registration_las` | 省略 | 绝对路径或相对输入目录的 LAS；默认唯一 `ResultOut/*.las` |
| `quality_max_median_px` | 2.0 | 整体及各镜头拟合重投影 median 上限 |
| `quality_max_p90_px` | 4.0 | 整体及各镜头拟合重投影 P90 上限 |
| `registration_min_support_fraction` | 0.25 | 可靠 LAS 表面支持比例下限 |
| `registration_max_median_m` | 0.1 | 支持区域点到面 median 上限 |
| `registration_max_p90_m` | 0.3 | 支持区域点到面 P90 上限 |

| 阶段 | 结果 |
| --- | --- |
| `prepare` | 输入、曝光和初始内参检查 |
| `masks` | 天空/人物语义掩码与预览 |
| `features` | 共享 LoMa 特征与检索描述子 |
| `matches` | 时间/检索候选、薄弱图补边与匹配缓存 |
| `observations` | 冻结有效观测、匹配索引及来源 |
| `calibration` | 自动支撑选择与四步受限标定 |
| `sfm` | 复用标定模型，全局建图、轨迹筛选和受限 BA |
| `finalize` | 发布并验收视觉模型，不重复 BA |
| `register` | LAS 配准、条件联合候选与交付选择 |
| `export` | COLMAP、掩码和 LAS 初始化 PLY，回读检查后发布 |

`--stage` 只执行指定阶段，上游必须已完成；新任务从 `prepare` 或默认 `all` 开始。算法和验收细节见[架构说明](architecture.md)。

## 续跑与导出

```bash
./run.sh --input input/场景 --work work/场景
./run.sh --work work/场景

# 单独执行阶段
.venv/bin/geosunpre run --work work/场景 --stage calibration

# 重新交付已验收结果；默认复制原图
.venv/bin/geosunpre export --work work/场景 --output /path/to/colmap --image-mode copy
.venv/bin/geosunpre check --output /path/to/colmap
```

工作目录冻结输入、配置、权重来源与默认交付选项；省略参数时继承。改变复制/链接、掩码、PLY 或体素参数时选择新 `--output`。`check --work` 检查任务默认输出。

默认附带掩码和 5 cm LAS 初始化 PLY。`--ply-voxel-size 0.1` 使用 10 cm，`--no-include-ply`、`--no-include-masks` 可省略对应产物；格式及迁移要求见[交付契约](colmap_contract.md)。

每个工作目录有进程锁，数值阶段原子发布；未提交模型在恢复时归档为 `*.interrupted-*`，完整阶段验证后复用。输入、缓存和权重按文件大小/修改时间及参数比较；运行环境版本只作记录。算法、输入或配置改变时使用新工作目录。当前配置/场景/任务协议为 **8 / 3 / 3**。

## 查看结果

| 路径，相对于 WORK | 内容 |
| --- | --- |
| `pipeline_status.json` | 阶段状态、失败原因、计时和资源 |
| `run_result.json` | 最近成功调用 |
| `calibration/selection.json` | 自动标定支撑与选择诊断 |
| `sfm/pair_protocol.json` | 匹配支持图及补边记录 |
| `sfm/masks/` | 语义有效图、原因标签和预览；导出时叠加初始 FOV |
| `registration/global/report.json` | 自动尺度、区域、秩与拟合诊断 |
| `registration/joint/unavailable.json` | 联合不可用或被跳过的原因，按需生成 |
| `registration/final/inspection.html` | 正式交付选择与质量概要 |
| `colmap/` | 最终训练数据、质量与来源清单 |

失败原因保存在阶段状态中。重投影和 LAS 残差来自拟合数据；验包证明交付结构一致性，独立精度和 3DGS 渲染质量需另行验证。
