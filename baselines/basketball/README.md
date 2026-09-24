# 篮球场开发基线

`baseline.json` 冻结当前采用的 **equal_control** 视觉相机解、原生鱼眼 COLMAP 输入和 LFS 配置的身份。数据文件仍位于 `work/basketball/`，这里仅保存相对路径、文件哈希及边界声明。这个基线供后续改进对照，并非最终图像一致性验收结果。

输入仅为具有同名 CameraPos 记录的 536 张 L/R 原图；pose 数值、LAS、M 不参与当前 SfM。L/R 各共享一组 `OPENCV_FISHEYE` 内参，每张曝光有独立位姿。LFS 使用共同掩码和共同视觉点云，536 张全部训练，没有测试集划分。人工标注 JSON 是粗标审阅材料，不是像素级真值。

```bash
uv run --no-sync geosunpre baseline-check --work work/basketball
uv run --no-sync geosunpre export-baseline --work work/basketball
```

`baseline-check` 默认核对清单、COLMAP 模型、图像成员、掩码和已有 LFS 导出身份；`--full` 额外重新计算全部原图与掩码的内容哈希。`export-baseline` 在校验后将单一等权分支打包到 `work/basketball/baseline_colmap/`，不运行 SfM、A/B 或 LFS 训练，也不覆盖现有实验输出。导出目录包含 `images/` 原图链接、独立拷贝的 `masks/`、`sparse/0/` 和冻结的 `lfs_config.json`。原图仍须保留在 `input/`。

模型来源和 A/B 证据见[全场 SfM](../../reports/basketball_global_sfm.md)、[联合 BA](../../reports/basketball_ground_joint_ba.md)、[LFS A/B](../../reports/basketball_lfs_ab.md)。改动模型、掩码、图像或配置时，应创建新的基线版本并明确晋升理由，不能覆盖本清单或原有目录。
