# GeosunPre 协作约定

个人使用的原生 C++17/CUDA 双鱼眼 + LAS 预处理工程，直接服务下游定制 LichtFeld Studio 3DGS 训练（参考 `configs/trainer.json`）。保持主线单一、简洁可改，算力与产物规模匹配训练器实际需求。

## 构建、验证与实验记录
- 编译使用 `tools/windows/build.ps1`（Ninja Release，保持 `/W4` 无新增警告），日常运行使用 `run.ps1`。
- `input/` 与 `models/` 只读；已有 `work/`、`output/` 成果目录不覆盖、不擅自删除。
- 按改动范围选择最小充分验证（编译、`check`、`--stage` 或 `export`）；主链路变更或明确要求时跑实景全流程，输出到 `work/<场景>-<后缀>` 等新目录并**保留在本地**供检查或下游试训。
- 关键实验的基线对比、指标变化与取舍结论统一追加记录在 `docs/experiments.md`（固定单文件维护，不零散新建多份临时报告），作为跨对话演进依据。

## 设计与契约
- 模块间直接传递 C++ 类型，JSON 仅用于配置、元数据和报告；几何计算用双精度，密集计算用 CUDA/TensorRT，图像、特征和 LAS 处理保持有界内存。
- 错误明确暴露，保留输入检查、几何验收与 `check` 回读校验，不用默认值或静默降级掩盖失败。
- 外部硬契约以 `docs/input.md`、`docs/models.md` 及训练器依赖的交付格式（`docs/output.md`）为准；内部流程与缓存细节见 `docs/reconstruction.md`、`docs/supervision.md`、`docs/cache.md`。
- 方案替换时直接删掉废弃代码、旧缓存字段和旁路 CLI 开关，不做历史包袱兼容，同步更新对应 `docs/`。
- 不主动做无关重构或安全审计；已确认方向直接推进；默认中文沟通，简述改动、验证结果与未验证项。
