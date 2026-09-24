# 工程主线与实验的代码边界

本项目的正式产物应是可识别、可验证、可交付给 LichtFeld-Studio 的视觉 SfM 相机解及 COLMAP 输入。实验用于检验候选，不应被正式入口隐式运行。当前保留的 `equal_control` 是**开发基线**，还不是通过高图像一致性验收的最终解。

## 三层代码边界

```text
src/geosunpre/
  *.py            # 共享相机/鱼眼投影、场景清单、特征、几何、BA 与验证部件
  pipeline/       # 全场前端与建图、基线校验、单分支 COLMAP 导出
  experiments/    # 局部/迁移对照、加权 A/B、LFS A/B 分析及各类 sfm_* probe
```

依赖只能从 `experiments → pipeline → 共享部件` 或 `experiments → 共享部件`。正式管线不能导入实验模块；实验可以读取已冻结的基线，但不能就地覆盖它。根命令 `geosunpre` 只在帮助中展示稳定流程和基线检查；`geosunpre-exp` 承载诊断和 A/B。旧 `geosunpre <实验命令>` 仍可使用，供已有报告复算。

迁移前，`global_reconstruction.py` 从 `transfer.py` 取配对自助抽样统计，而该文件又承载第三片段试验。现已把统计移到共享 `statistics.py`，把全场建图放入 `pipeline/`，并拆开局部实验执行器与通用前端、BA、验证函数。现有已采用的等权**产物**由独立清单和只读校验固定。新增的 `build-baseline` 入口在 `pipeline/` 内独立完成局部视觉标定、全场 SfM、等权 BA 和单分支导出，不导入实验模块，也不读取静态点人工标注。近地点加权联合 BA 仍留在实验代码中。

## 基线以小清单冻结，不复制大数据

```text
baselines/basketball/
  baseline.json       # 纳入 Git；相对路径、状态、关键哈希、来源协议
  README.md           # 输入边界、运行/校验/导出方法、已知限制
experiments/basketball/
  README.md           # 实验索引与晋升结论；历史报告和冻结协议的入口
work/basketball/      # 大型数据与产物，仍在原位置，不进入 Git
```

`baseline.json` 已记录：仅同名 pose 记录的 536 张 L/R、每曝光独立位姿；原生 `OPENCV_FISHEYE`；输入 pose 数值、LAS、M 均未进入几何拟合；当前模型 `global_sfm/ground_joint_ba/equal_control/sparse/0` 的文件哈希；LFS 等权 COLMAP 导出、共同掩码/点云和配置的协议指纹；“开发基线、非最终验收”的状态。来源是已保存的 `global_sfm/lfs_ab/protocol.json`。用户提供的粗标 JSON 保留原件，但不写成像素级真值。

单分支 `export-baseline` 先核对清单，再把已冻结等权输入打包到 `work/basketball/baseline_colmap/`：模型、掩码与点云单独拷贝，原图用链接保持不重采样。`lfs-export-ab` 继续留在实验入口。打包不运行 A/B 或 LFS 训练；当前 536 张全部训练、不划测试集的约定保持不变。

## 实验记录如何分开

每个实验有唯一标识、冻结输入/协议、命令、产物位置、评估口径及结论，结论只取“晋升基线 / 未晋升 / 仅诊断”。[报告索引](../reports/README.md)与[历史复算记录](../experiments/basketball/HISTORY.md)保留旧试验；后续新实验直接写入 `experiments/basketball/`。历史报告暂不批量移动：大量链接指向 `work/`，还有脚本固定读取旧目录、特征目录保存绝对路径。需要移动时同时改路径、核对哈希和全部链接，保留旧复算入口或明确版本迁移。

基线晋升应是显式操作：冻结输入、模型、掩码和配置身份；说明相对于旧基线的原图静态结构与近地面检查、LFS 训练视图对比及退化项；最后更新基线清单。指标的局部改善、注册率或训练残差单独都不等于高一致性验收，也不自动触发晋升。

## 实施顺序

1. 已为重组前的源码、配置、文档建立初始 Git 提交 `6a4ea49`；关键等权产物的相对路径与哈希写入基线清单。`input/`、`work/` 仍需独立备份。
2. 已增加只读 `baseline-check` 和单分支 `export-baseline`，并实测当前 536 张模型和全部原图/掩码内容哈希。
3. 已抽离共享函数，迁移实验模块与命令，保留旧命令兼容；正式管线不导入实验模块。
4. 历史 `work/` 产物仍需逐项归档。大型 LFS 快照、人工标注、默认模型和报告证据在可验证归档前不删除。
5. 新增 `build-baseline` 和 `check-computed-baseline`，把从原图到 LFS COLMAP 输入的等权流程与独立身份校验接通；历史冻结清单不变。完整 536 张 SfM 尚未重算，当前验证范围是全部单元测试、小型合成 COLMAP 打包、历史点云和配置内容对照，以及真实原图的图像选择阶段。

本次重组及后续自动构建入口工作没有重新求解完整 SfM 或训练 LFS；现有单分支目录仍是对已冻结等权输入的打包副本。
