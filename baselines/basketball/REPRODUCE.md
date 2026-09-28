# 篮球场基线复跑

## 从原图构建 COLMAP 数据集

选择新的工作目录，保留历史冻结产物。

```bash
uv sync --locked --extra loma --extra segmentation
uv run --no-sync geosunpre build-baseline \
  --input input/篮球场 --work work/basketball_rerun_01
```

依次完成同名 CameraPos 的 L/R 图像选择、全局 SegFormer 与设备遮挡掩码、LoMa 特征与匹配、全局留出冻结、70–94 帧局部视觉标定、独立曝光 SfM、40 次迭代上限的等权鱼眼 BA 和 COLMAP 导出。pose 数值、LAS、M、人工静态点标注不参与求解。当前构建仍要求篮球场的 536 张图，并非已完成任意场景配置化。

求解模型位于 `global_sfm/baseline_equal/sparse/0/`，交付目录为 `WORK/colmap/`。默认包含五份 COLMAP 模型文件、原图链接、`colmap_export.json` 与 `quality.json`。不生成 LFS 配置，不设训练/测试划分，也不要求先完成下游 A/B 实验。

跨机器交付加 `--image-mode copy`。`--include-masks` 和 `--include-ply` 分别附带组合有效域掩码与全部稀疏点的 PLY，两者均可省略。详情见[输出契约](../../docs/colmap_contract.md)。

## 分阶段运行与独立检查

`--stage prepare|global-data|holdout|local-calibration|global-sfm|equal-ba|export` 选择单个阶段；后续阶段要求上游产物已存在。局部标定直接使用全局特征编号和留出规则，必须先完成 `holdout`。

```bash
uv run --no-sync geosunpre build-baseline \
  --work work/basketball_rerun_01 --stage equal-ba
uv run --no-sync geosunpre build-baseline \
  --work work/basketball_rerun_01 --stage export \
  --output work/basketball_rerun_01/colmap_delivery --image-mode copy
uv run --no-sync geosunpre check-colmap \
  --output work/basketball_rerun_01/colmap_delivery
```

导出结束会自动校验内容和模型内部一致性。检查拷贝包不依赖原工作目录。`global_sfm/comparison.json` 和 `baseline_equal/validation_summary.json` 是同次采集的观测留出诊断；新协议在局部标定前冻结全局留出，贯穿局部及全局求解排除。这仍不代表独立采集验收。旧流程的模型不能继续用于新协议求解，详见[跨阶段留出协议](../../docs/unified_holdout.md)。

新构建仍未对全部 536 张图完整重跑，已有验证覆盖小型合成模型、真实评估函数、导出与迁移。完整构建的重建质量及资源需求需要首次完整运行后评估。

## 历史基线审计与导出

```bash
uv run --no-sync geosunpre baseline-check --work work/basketball --full
uv run --no-sync geosunpre export-baseline --work work/basketball
uv run --no-sync geosunpre check-colmap --output work/basketball/colmap
```

`baseline-check` 继续审计历史 `baseline.json` 指定的模型、LFS 配置及实验产物；它与新数据集的独立校验承担不同职责。`export-baseline` 核对冻结场景清单与源模型后输出通用 `WORK/colmap/`，不读取 LFS 配置或 LFS 初始化点云。该导出记录本次原图内容身份；需要核对原图是否仍符合完整历史实验身份时，使用前面的 `baseline-check --full`。

历史 `WORK/baseline_colmap/`、原始清单及 LFS 配置保留为旧实验记录。历史 LFS 训练命令见[实验记录](../../experiments/basketball/HISTORY.md)，不属于正式构建步骤。
