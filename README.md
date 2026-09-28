# GeosunPre

将原始采集数据预处理为 **COLMAP 格式的数据集**。当前重建流程支持 Geosun 异步 L/R 双鱼眼：每次曝光独立求位姿，每个镜头共享一组 `OPENCV_FISHEYE` 内参。正式输出保留原图、相机、位姿、三维点与观测轨迹；不包含下游训练配置，也不决定训练/测试划分。

当前完整构建仍采用篮球场配方：536 张具有同名 CameraPos 记录的 L/R 原图，70–94 帧局部视觉标定。输入 pose 数值、LAS 和 M 图像不参与几何拟合；`.pro` 内参是初始候选。已保存的 `equal_control` 是开发基线，尚未通过最终几何质量验收。

新流程先完成全局特征与匹配，再冻结留出集，随后执行局部标定和全局重建；相同留出观测贯穿各阶段排除。旧标定结果不能直接复用为新协议结果，详见[跨阶段留出协议](docs/unified_holdout.md)。

## 构建与导出

```bash
uv sync --locked --extra loma --extra segmentation
uv run --no-sync geosunpre build-baseline \
  --input input/篮球场 --work work/basketball_rerun_01
```

默认输出 `work/basketball_rerun_01/colmap/`，原图使用相对符号链接。跨机器交付使用 `--image-mode copy`；可用 `--include-masks`、`--include-ply` 附带掩码或完整稀疏点云的 PLY。PLY 不执行下游初始化筛选，不要求至少 10,000 个点。完整 SfM 涉及 GPU 与模型下载，新构建入口尚未对全部 536 张图重新运行。

已有模型可直接导出，无需运行重建：

```bash
uv run --no-sync geosunpre export-colmap \
  --scene work/basketball/global_sfm/manifest.json \
  --model work/basketball/global_sfm/ground_joint_ba/equal_control/sparse/0 \
  --output work/basketball/colmap_delivery --image-mode copy
uv run --no-sync geosunpre check-colmap --output work/basketball/colmap_delivery
```

导出前在临时目录完成文件及模型校验，成功后发布整个目录。已有输出仅在内容与选项完全一致时复用。`check-colmap` 只依赖导出包及链接图像；拷贝模式下，校验不依赖原 `input/`、`work/` 或特征缓存。格式与内容校验通过不代表几何质量已通过验收。

## 历史基线

[基线清单](baselines/basketball/README.md)记录既有视觉模型及历史 LFS 对照身份：

```bash
uv run --no-sync geosunpre baseline-check --work work/basketball --full
uv run --no-sync geosunpre export-baseline --work work/basketball
```

`baseline-check` 保留历史模型、LFS 配置与实验产物的只读审计。`export-baseline` 核对冻结场景清单和源模型后，输出新的通用 `work/basketball/colmap/`，不读取 LFS 配置。历史 `baseline_colmap/` 及冻结清单保留；新的导出身份为 `colmap_export.json`。

## 代码与验证

- `src/geosunpre/`：相机、特征、几何、BA 和评估部件。
- `src/geosunpre/pipeline/`：构建、COLMAP 导出、独立数据集校验及历史基线审计。
- `src/geosunpre/experiments/`：局部/迁移诊断、加权 A/B 和下游 LFS 试验，使用 `geosunpre-exp`。
- [输出契约](docs/colmap_contract.md)、[工程边界](docs/engineering_structure.md)、[基线复跑](baselines/basketball/REPRODUCE.md)。
- [实验报告](reports/README.md)、[历史命令](experiments/basketball/HISTORY.md)保留已有证据。

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q tests
```

自动验证覆盖几何数值约定、跨阶段留出与身份检查、导出回读、损坏检测、包迁移和失败发布。新隔离协议尚未完整重建实测；场景配置化及完整流水线恢复仍需后续改造，见工程边界文档。
