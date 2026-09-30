# 验证摘要索引

JSON 保存可追溯的小型结果与原始记录身份。完整日志和模型位于本地 `work/`，
不随 Git 分发；摘要中的路径在其他机器上可能不存在。

| 摘要 | 模式与范围 | 阅读说明 |
| --- | --- | --- |
| [basketball_joint_pipeline_20260930.json](basketball_joint_pipeline_20260930.json) | 默认 production + local + joint 完整冷启动及缓存续跑 | [生产记录](../production_pipeline.md#默认联合流程完整运行验证2026-09-30)：536 图、75,835 点；4 个 fit 瓦片退化触发已验收全局配准回退，LAS 坐标交付；137 项测试和 8,765 个冻结文件复用核对通过 |
| [basketball_las_alignment_20260930.json](basketball_las_alignment_20260930.json) | 既有纯图像模型与 LAS 的重建后配准诊断 | [配准报告](../las_alignment.md)：全局候选、空间留出与方向差异；不代表绝对精度或整场配准验收 |
| [basketball_las_structure_20260930.json](basketball_las_structure_20260930.json) | 冻结配准候选的结构与低支持原因诊断 | [结构检查](../las_alignment.md#进一步检查结构与低支持区域)：逐点重现、原图定位、LAS 时段差异与观察侧歧义 |
| [basketball_las_refinement_20260930.json](basketball_las_refinement_20260930.json) | 可靠结构参与的全局细化探索 | [结构细化](../las_alignment.md#基于可靠结构继续细化)：竖直结构中位误差下降、P90 增加，整体收益未明确；保留参考候选 |
| [basketball_joint_alignment_20260930.json](basketball_joint_alignment_20260930.json) | 固定内参、图像主导的 pose / LAS 联合 BA 对照 | [联合 BA 实跑](../las_alignment.md#固定内参联合-ba-实跑)：图像回归门槛通过，位姿变化很小，LAS 整体精度收益未明确；未训练 3DGS |
| [industrial_pipeline_20260930.json](industrial_pipeline_20260930.json) | 一键启动、缓存续跑与 production 导出恢复 | [流程验证](../engineering_structure.md#当前流程验证2026-09-30)：89 项测试通过，536 张图验包通过，81 个冻结产物未改写；不包含新版本冷启动计时 |
| [basketball_production_20260929.json](basketball_production_20260929.json) | 2026-09-29 production，全流程冷启动 | [生产报告](../production_pipeline.md)：536 张全部注册，20:31.30，未达到 20 分钟目标 |
| [calibration_ablation_20260929.json](calibration_ablation_20260929.json) | validation，复用前端的 local/input 几何对照 | [消融报告](../calibration_ablation.md)：支持保留局部标定；耗时不是完整冷启动 |
| [basketball_resources_20260929.json](basketball_resources_20260929.json) | 历史 validation，资源优化 | [资源报告](../resources.md)：峰值 RSS 4.42 GiB；受并行训练影响，不比较耗时 |
| [basketball_20260929.json](basketball_20260929.json) | 历史 validation，性能优化 | [性能报告](../performance.md)：单次冷启动 19:51.30，不代表当前生产耗时或稳定上限 |

生产报告的残差来自参与拟合的观测，历史验证报告包含留出预测误差，两者不能直接比较。
交付校验只证明包内内容与来源记录一致，不证明独立几何精度或下游渲染质量。
历史摘要保留当次源码、配置和产物身份；后续修改文档不回写历史运行记录。
