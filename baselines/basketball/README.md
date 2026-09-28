# 篮球场开发基线

`baseline.json` 冻结现有 **equal_control** 视觉相机解及历史 LFS 对照的身份。数据仍位于 `work/basketball/`，这里保存相对路径、哈希和边界声明。`lfs_config.json` 是历史实验记录，不是正式 COLMAP 导出的依赖。

输入为具有同名 CameraPos 记录的 536 张 L/R 原图；pose 数值、LAS、M 不参与 SfM。L/R 各共享一组 `OPENCV_FISHEYE` 内参，每次曝光独立求位姿。人工标注为粗标审阅材料，不能当作精确像素真值。

```bash
uv run --no-sync geosunpre baseline-check --work work/basketball --full
uv run --no-sync geosunpre export-baseline --work work/basketball
uv run --no-sync geosunpre check-colmap --output work/basketball/colmap
```

`baseline-check` 只读核对历史清单、模型、图像成员、掩码、LFS 配置和既有导出身份；`--full` 额外计算完整的原图/掩码内容身份。它保留旧实验的审计口径。

`export-baseline` 核对冻结场景清单和源模型后，输出通用的 `WORK/colmap/`，包含原图链接、完整 COLMAP 模型及 `colmap_export.json`。无需 LFS 配置或历史筛选 PLY。`--image-mode copy` 用于可迁移交付；掩码和 PLY 分别通过 `--include-masks`、`--include-ply` 附带。导出记录本次原图内容哈希；核对原图与完整历史实验身份的一致性使用 `baseline-check --full`。

历史 `WORK/baseline_colmap/` 及清单保留。新的 COLMAP 数据集使用 `check-colmap` 独立校验。原来的 LFS 全图训练、共同掩码/点云、无测试集约定仅属于历史下游 A/B。

[复跑说明](REPRODUCE.md)提供从原图构建和分阶段执行方法。模型来源及已有质量证据见[全场 SfM](../../reports/basketball_global_sfm.md)、[联合 BA](../../reports/basketball_ground_joint_ba.md)、[LFS A/B](../../reports/basketball_lfs_ab.md)。本基线仍未通过最终几何质量验收，新改进应建立独立版本并说明晋升依据。
