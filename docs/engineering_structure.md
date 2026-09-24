# 工程主线与实验的组织方案

本项目的正式产物应是可识别、可验证、可交付给 LichtFeld-Studio 的视觉 SfM 相机解及 COLMAP 输入。实验用于检验候选，不应被正式入口隐式运行。当前保留的 `equal_control` 是**开发基线**，还不是通过高图像一致性验收的最终解。

## 三层代码边界

```text
src/geosunpre/
  core/           # 相机/鱼眼投影、场景清单、特征与几何、BA 通用部件
  pipeline/       # 输入选择、全场前端与建图、基线校验、单分支 COLMAP 导出
  experiments/    # 局部/迁移对照、加权 A/B、LFS A/B 分析及各类 sfm_* probe
```

依赖只能从 `experiments → pipeline → core` 或 `experiments → core`。正式管线不能导入实验模块；实验可以读取已冻结的基线，但不能就地覆盖它。根命令 `geosunpre` 只暴露稳定流程和基线检查；另设 `geosunpre-exp` 承载诊断和 A/B。迁移期间保留旧命令别名，直到报告中的复算命令全部改完并核对无误。

这不是按现有文件名直接移动。目前 `global_reconstruction.py` 导入 `local_data.py`、`local_reconstruction.py`、`transfer.py` 和 `track_consistency.py` 的函数；`ground_joint_ba.py` 又从 `ground_pose_probe.py` 借用几何与评估函数。前者要把真正通用的前端、BA、评估函数抽到 `core/`，后者要将等权模型的生成/校验与近地点实验对照分离。否则只是换目录，正式流程仍依赖实验代码。

## 基线以小清单冻结，不复制大数据

```text
baselines/basketball/
  baseline.json       # 纳入 Git；相对路径、状态、关键哈希、来源协议
  README.md           # 输入边界、运行/校验/导出方法、已知限制
experiments/basketball/
  README.md           # 实验索引与晋升结论；历史报告和冻结协议的入口
work/basketball/      # 大型数据与产物，仍在原位置，不进入 Git
```

`baseline.json` 应明确记录：仅同名 pose 记录的 536 张 L/R、每曝光独立位姿；原生 `OPENCV_FISHEYE`；输入 pose 数值、LAS、M 均未进入几何拟合；当前模型 `global_sfm/ground_joint_ba/equal_control/sparse/0` 的文件哈希；LFS 等权 COLMAP 导出、共同掩码/点云和配置的协议指纹；“开发基线、非最终验收”的状态。现有 `global_sfm/lfs_ab/protocol.json` 已保存图像/掩码汇总哈希和两组模型文件哈希，可作为建立清单的来源。用户提供的粗标 JSON 保留原件，但不写成像素级真值。

正式导出应提供单分支 `export-baseline`：先核对清单，再复制/链接已冻结的等权模型、图像、掩码和共同点云，输出 LFS 所需的 COLMAP 目录。现有 `lfs-export-ab` 同时生成两组输入和比较配置，保留在实验入口。正式导出不能以运行 A/B 为前提，也不应要求重新训练 LFS。当前 536 张全部训练、不划测试集的约定保持不变。

## 实验记录如何分开

每个实验有唯一标识、冻结输入/协议、命令、产物位置、评估口径及结论，结论只取“晋升基线 / 未晋升 / 仅诊断”。先用[现有报告索引](../reports/README.md)归类；后续新实验直接写入 `experiments/basketball/`。历史报告暂不批量移动：本次盘点有 273 个 Markdown 链接指向 `work/`，还有脚本固定读取旧目录、特征目录保存绝对路径。需要移动时同时改路径、核对哈希和全部链接，保留旧复算入口或明确版本迁移。

基线晋升应是显式操作：冻结输入、模型、掩码和配置身份；说明相对于旧基线的原图静态结构与近地面检查、LFS 训练视图对比及退化项；最后更新基线清单。指标的局部改善、注册率或训练残差单独都不等于高一致性验收，也不自动触发晋升。

## 实施顺序

1. 把现有源码、配置、文档建立 Git 基线；保存关键产物的相对路径和哈希清单。`input/`、`work/` 继续不进 Git，但需要独立备份。
2. 增加只读的 `baseline-check`，验证图像成员、相机模型、模型/导出协议哈希和链接目标；再增加独立的单分支 `export-baseline`。
3. 抽离共享函数，消除 `pipeline → experiments` 依赖；随后迁移实验模块与命令，同时保持旧命令兼容并核对测试/复算入口。
4. 最后整理历史报告与 `work/`，逐项归档未晋升产物。大型 LFS 快照、人工标注、默认模型和报告证据在可验证归档前不删除。

本文件是工程边界和迁移顺序，尚未改变现有命令、代码路径或任何 SfM/LFS 产物。
