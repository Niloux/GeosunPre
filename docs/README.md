# 文档索引

一键入口为 `./run.sh`，正式命令为 `run / export / check`。新运行默认 **production + local + joint**：保留局部标定，
使用通过质量筛选的全部图像观测，然后完成 LAS 配准和固定内参联合 BA，自动交付验收后的 LAS 坐标模型。
图像留出只在显式选择的 validation 模式中执行；纯 BA 是联合优化的必要回归对照。

## 使用与维护

| 文档 | 内容 |
| --- | --- |
| [项目首页](../README.md) | 安装、一键启动、权重缓存、输入要求与场景配置 |
| [生产与验证模式](production_pipeline.md) | 模式边界、续跑身份、生产质量检查及完整实跑结果 |
| [工程结构](engineering_structure.md) | 模块职责、工作目录、阶段恢复与数值流程 |
| [COLMAP 交付契约](colmap_contract.md) | 交付内容、相机模型、来源记录与独立校验 |
| [LAS 配准与联合优化](las_alignment.md) | 默认配准验收、回退、坐标与输入约定，以及独立结构诊断和三方案对照 |
| [仓库管理](repository.md) | 目录职责、配置兼容与证据保留范围 |

## 研发协议与历史记录

| 文档 | 范围 |
| --- | --- |
| [统一留出协议](unified_holdout.md) | 仅 validation：拟合前冻结留出、预测评估与可比性 |
| [局部标定消融](calibration_ablation.md) | validation 下 local/input 对照；保留局部标定的依据 |
| [性能记录](performance.md) | 历史 validation 优化及完整计时 |
| [资源记录](resources.md) | 历史 validation 内存与显存验证 |
| [验证摘要索引](benchmarks/README.md) | 随仓库保存的 JSON 结果、来源和比较限制 |

2026-09-29 的生产完整冷启动记录为 20 分 31.30 秒，尚未达到 20 分钟目标。
该历史计时不代表新增联合流程。早期启动与恢复验证见
[流程验证摘要](benchmarks/industrial_pipeline_20260930.json)。
2026-09-30 的联合流程已完整实跑并完成独立校验和续跑，最终因 4 个局部瓦片退化采用
已验收的全局配准回退，交付仍为 LAS 坐标；统一流程时钟 27:45.13，GNU time 26:54.56，
均未达到 20 分钟目标。详见 [当前联合流程记录](benchmarks/basketball_joint_pipeline_20260930.json)。
生产拟合残差与验证留出预测误差采用不同口径，不能直接比较。
