# 验证摘要索引

JSON 保存可追溯的小型结果与原始记录身份。完整日志和模型位于本地 `work/`，
不随 Git 分发；摘要中的路径在其他机器上可能不存在。

| 摘要 | 模式与范围 | 阅读说明 |
| --- | --- | --- |
| [industrial_pipeline_20260930.json](industrial_pipeline_20260930.json) | 一键启动、缓存续跑与 production 导出恢复 | [流程验证](../engineering_structure.md#当前流程验证2026-09-30)：89 项测试通过，536 张图验包通过，81 个冻结产物未改写；不包含新版本冷启动计时 |
| [basketball_production_20260929.json](basketball_production_20260929.json) | 2026-09-29 production，全流程冷启动 | [生产报告](../production_pipeline.md)：536 张全部注册，20:31.30，未达到 20 分钟目标 |
| [calibration_ablation_20260929.json](calibration_ablation_20260929.json) | validation，复用前端的 local/input 几何对照 | [消融报告](../calibration_ablation.md)：支持保留局部标定；耗时不是完整冷启动 |
| [basketball_resources_20260929.json](basketball_resources_20260929.json) | 历史 validation，资源优化 | [资源报告](../resources.md)：峰值 RSS 4.42 GiB；受并行训练影响，不比较耗时 |
| [basketball_20260929.json](basketball_20260929.json) | 历史 validation，性能优化 | [性能报告](../performance.md)：单次冷启动 19:51.30，不代表当前生产耗时或稳定上限 |

生产报告的残差来自参与拟合的观测，历史验证报告包含留出预测误差，两者不能直接比较。
交付校验只证明包内内容与来源记录一致，不证明独立几何精度或下游渲染质量。
历史摘要保留当次源码、配置和产物身份；后续修改文档不回写历史运行记录。
