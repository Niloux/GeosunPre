# 生产模式与验证模式

## 当前默认

新运行默认 `pipeline_mode = "production"`、`calibration_mode = "local"`。
保留局部标定，以及全局固定内参 BA、三图支持轨迹筛选、共享内参 BA 和最终 BA。
生产模式使用全部通过质量筛选的观测，不划分留出集，不按比例排除可用轨迹。

| 行为 | production（默认） | validation |
| --- | --- | --- |
| 局部视觉标定 | 保留 | 保留 |
| 几何拟合前冻结来源 | 全量观测身份 | 统一留出协议 |
| 留出轨迹 | 无 | 符合条件的原始轨迹中抽取 20% |
| 每阶段留出预测评估 | 无 | 保留 |
| 共同支持查询与 bootstrap | 无 | 保留 |
| 掩码、两视图几何验证、轨迹筛选 | 保留 | 保留 |
| 深度、鱼眼单调性与覆盖检查 | 保留 | 保留 |
| 最终指标 | 拟合观测重投影残差与健康诊断 | 原始/高置信留出预测误差 |
| COLMAP 导出及独立校验 | 保留 | 保留 |

生产模式中的“全部”不意味着保留错配；掩码、匹配几何验证、旋转一致性和三图支持筛选仍然生效。
拟合观测的残差不是测试集精度，不能与此前留出 median/P90 直接比较并宣称质量提高。

## 运行

```bash
uv run --no-sync geosunpre run \
  --input input/篮球场 --work work/basketball_production_fresh \
  --config configs/basketball.toml --include-masks
```

`run` 已内置交付校验。续跑只需 `geosunpre run --work ...`；输出路径和导出选项读取冻结任务。

仅复用权重时可显式传入 `--model-cache /path/to/huggingface/cache`。
要执行研发对照，在新的 TOML 中设置 `pipeline_mode = "validation"`，使用新工作目录。
`calibration_mode` 是独立选项；正式默认 local，input 初始化仅用于显式消融。

生产阶段：

`prepare → masks → features → matches → observations → calibration → sfm → finalize → export`

验证模式将 `observations` 替换为 `holdout`。调用另一模式专属阶段会报错。

## 身份与续跑

配置版本 4 必须保存 `pipeline_mode`。版本 1/2/3 的旧工作目录解释为 validation，
旧配置和已有几何协议不改写，续跑时省略 `--config`。
切换模式或修改配置须使用新工作目录，不能把旧留出模型重新标记为生产模型。

生产模式在 `sfm/observation_protocol.json` 冻结前端缓存内容哈希、场景、配对和配置身份，
声明为验证排除的观测数为 0；不生成 `split.json`、`triangle_validation_split.json` 或
`holdout_protocol.json`。几何部件接收内存中的空排除列表，全部合法匹配进入原有几何筛选。

生产与验证模式的协议互斥。观测来源改变、匹配缓存修改、混入另一模式的划分文件，都会拒绝继续。
这些身份检查贯穿局部标定、全局建图、最终 BA 与正式导出。

## 交付与诊断

生产最终模型仍位于 `sfm/final/sparse/0`；质量报告为 `sfm/final/quality_summary.json`，
复制到交付包 `quality.json`。包含注册数、点数、观测数、每图覆盖、拟合观测重投影残差、
正深度及鱼眼单调性检查、内参及触界记录。最终生产模型还必须保持连通，
每图至少 100 条观测并覆盖 8×8 网格中的至少 12 个网格，与既有轨迹筛选要求一致。

验证模式继续使用 `sfm/final/validation_summary.json`。
生产拟合残差门槛由 `quality_max_median_px`、`quality_max_p90_px` 配置，对整体及 L/R 分别检查；
`quality_bounds_policy` 可选 `warn` 或 `reject`。篮球场新配方使用 2 / 4 px 和 reject，
通用默认不设置残差上限、触界告警；所有生产结果均拒绝空或非有限残差。
这些是交付操作门槛，不是独立几何精度。失败会阻止导出，不自动修改阈值。
两种模式均输出 `colmap_export.json` 来源记录，生产来源明确记录 mode 和零留出排除。
`check` 校验交付内容完整性，不重新计算几何精度。

## 完整运行验证

以下记录对应 2026-09-29 当次源码与冻结配置。当前任务契约、恢复、缓存复用与交付门槛改进
尚未重跑完整冷启动，不能把该计时作为新版本的性能成绩。

本轮使用新目录 `work/basketball_production_20260929_01`，只复用预训练权重，
掩码、特征、匹配及全部几何重新计算；包含原图复制、掩码导出及交付校验。
预先冻结源码哈希与验收项，记录位于 `work/production_validation_20260929/`。
本轮已完成，验收项全部通过；完整冷启动耗时 **20 分 31.30 秒**，未达到 20 分钟目标。
这次切换采用的依据是生产/验证职责分离及完整运行通过，不是速度提升或独立几何精度改善。

| 项目 | 结果 |
| --- | ---: |
| 注册图像 | 536 / 536 |
| 最终稀疏点 | 75,893 |
| 最终拟合观测 | 482,968 |
| 为验证排除的观测 | 0 |
| 无效深度 | 0 |
| 每图最少观测 | 244 |
| 每图最少占用 8×8 网格 | 19 |
| 相机 160° 单调性 | L/R 均通过 |
| 内参触界 | 无 |
| 拟合残差 median / P90 | 1.07247 / 2.34930 px |
| 进程峰值 RSS | 约 4.63 GiB |
| 独立交付校验 | 通过 |

上述残差来自参与拟合的观测，不是原先留出分数的替代品。
本轮没有独立采集几何验收或下游 3DGS 训练/渲染比较。

| 阶段 | 耗时 / 秒 |
| --- | ---: |
| prepare | 0.23 |
| masks | 138.44 |
| features | 92.29 |
| matches | 96.51 |
| observations | 3.03 |
| calibration | 29.77 |
| sfm | 858.68 |
| finalize | 15.10 |
| export | 27.62 |

阶段求和与完整进程耗时之间包含启动等开销。全量特征与匹配缓存命中均为 0。
对话中断时外层采集会话丢失，但实际流水线持续持有工作目录锁、使用 GPU 并未重启；
GNU time 保留完整 20:31.30 计时和退出码 0。完成后补写汇总并再次独立验证交付包。
计时含流程内导出与校验；额外独立校验约 3.58 秒，单独记录，不计入 20:31.30。

验证证据：

- 72 项测试通过，新增模式分发、零排除、观测身份保护、旧版本兼容、无留出评估和生产交付检查。
- 两个既有验证工作目录的冻结几何协议核对通过，文件哈希未改变。
- 生产运行实际未生成 split、holdout、validation、comparison 或 bootstrap 产物。
- 运行期间源码与正式配置哈希保持不变。
- 可追溯摘要：[basketball_production_20260929.json](benchmarks/basketball_production_20260929.json)。
- 原始证据位于 `work/production_validation_20260929/`；交付包位于
  `work/basketball_production_20260929_01/colmap/`。
