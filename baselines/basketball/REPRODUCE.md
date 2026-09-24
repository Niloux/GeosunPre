# 篮球场基线复跑

## 从原始图像完整构建

在仓库根目录运行。选择新的工作目录，保留现有 `work/basketball` 冻结产物。

```bash
uv sync --locked --extra loma --extra segmentation
uv run --no-sync geosunpre build-baseline \
  --input input/篮球场 \
  --work work/basketball_rerun_01
```

这一条命令依次完成：精确同名 CameraPos 的 L/R 图像选择、70–94 帧局部视觉内参标定、SegFormer 与固定设备遮挡规则生成掩码、LoMa 全场特征和匹配、全场独立曝光 SfM、40 次迭代上限的等权鱼眼联合 BA，以及单分支 LichtFeld-Studio COLMAP 导出。原始 pose **仅用于确定图像是否入选**，其数值和 LAS 不进入求解；M 和无同名 pose 的图像不入选。整个构建不读取静态点人工标注、不运行近地面探针，也不要求先跑加权 A/B。固定设备遮挡多边形仍是版本化掩码规则，无需逐次人工画图。

输出模型在 `work/basketball_rerun_01/global_sfm/baseline_equal/sparse/0/`，LFS 输入在 `work/basketball_rerun_01/baseline_colmap/`。后者包含原生鱼眼 `sparse/0/`、原图链接、组合有效域掩码、视觉点云、`lfs_config.json` 和独立的 `computed_baseline.json` 身份清单。点云距离筛选按本次视觉轨迹尺度自动设定，不依赖旧重建的世界尺度。LFS 配置纳入本仓库的 `baselines/basketball/lfs_config.json`，全部 536 张图训练，不划测试集。

导出结束时会自动核对一次全部内容；之后也可只读复核：

```bash
uv run --no-sync geosunpre check-computed-baseline --work work/basketball_rerun_01
```

可用 `--stage prepare|local-calibration|global-data|global-sfm|equal-ba|export` 单独执行某一阶段，检查或续跑时保持同一个 `--work`。例如只重新核对等权 BA 的保存模型及自动留出评估：

```bash
uv run --no-sync geosunpre build-baseline \
  --work work/basketball_rerun_01 --stage equal-ba
```

`global_sfm/comparison.json` 与 `global_sfm/baseline_equal/validation_summary.json` 保存没有人工标签参与的轨迹留出检查。它们用于发现回归，不能当作像素级真值或高一致性 3DGS 的最终验收。新复算是独立候选，其二进制模型或哈希不要求等于历史冻结版本；现有 `baseline.json` 不会被覆盖。

如需在导出的数据上单独训练 LFS，使用新的训练输出目录：

```bash
/home/wuyou/LichtFeld-Studio/build/train/lfs-train \
  -d work/basketball_rerun_01/baseline_colmap \
  -o work/basketball_rerun_01/lfs_train_01 \
  --config work/basketball_rerun_01/baseline_colmap/lfs_config.json \
  --no-download
```

完整 SfM 会消耗 GPU、模型下载、磁盘和较长时间。当前实现完成了代码和小型合成模型验证，尚未对原始 536 张图重新运行，因此新构建的实际重建质量仍需在第一次完整运行后检查。

## 核对或重新打包历史冻结基线

这两条命令只处理现有 `work/basketball` 产物，不运行 SfM：

```bash
uv run --no-sync geosunpre baseline-check --work work/basketball --full
uv run --no-sync geosunpre export-baseline --work work/basketball
```

它们继续使用历史 `baselines/basketball/baseline.json` 的固定身份，并输出 `work/basketball/baseline_colmap/`。新构建的 `computed_baseline.json` 则记录本次图像、掩码、模型和 LFS 配置的内容身份，不能用历史 `baseline-check` 核对。
