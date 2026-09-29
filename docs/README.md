# 文档索引

当前正式入口为 `run / export / check`。新运行默认 **production + local**：保留局部标定，
使用通过质量筛选的全部观测；留出划分和方案对照只在显式选择的 validation 模式中执行。

## 使用与维护

| 文档 | 内容 |
| --- | --- |
| [项目首页](../README.md) | 安装、输入要求、正式命令与场景配置 |
| [生产与验证模式](production_pipeline.md) | 模式边界、续跑身份、生产质量检查及完整实跑结果 |
| [工程结构](engineering_structure.md) | 模块职责、工作目录、阶段恢复与数值流程 |
| [COLMAP 交付契约](colmap_contract.md) | 交付内容、相机模型、来源记录与独立校验 |
| [仓库管理](repository.md) | 目录职责、配置兼容与证据保留范围 |

## 研发协议与历史记录

| 文档 | 范围 |
| --- | --- |
| [统一留出协议](unified_holdout.md) | 仅 validation：拟合前冻结留出、预测评估与可比性 |
| [局部标定消融](calibration_ablation.md) | validation 下 local/input 对照；保留局部标定的依据 |
| [性能记录](performance.md) | 历史 validation 优化及完整计时 |
| [资源记录](resources.md) | 历史 validation 内存与显存验证 |
| [验证摘要索引](benchmarks/README.md) | 随仓库保存的 JSON 结果、来源和比较限制 |

生产完整冷启动记录为 20 分 31.30 秒，尚未达到 20 分钟目标。
生产拟合残差与验证留出预测误差采用不同口径，不能直接比较。
