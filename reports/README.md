# 篮球场项目结构与实验索引

盘点时间：2026-09-24。本页区分当前使用的视觉模型、对照候选和只用于定位问题的诊断；下面的目录大小是 `du` 的约数。**目前没有经高图像一致性验收的最终 SfM 解。** 输入 pose、LAS、`.pro` 内参均为先验，不是真值；只有具备同名 pose 记录的 536 张 L/R 图像参加当前 SfM，曝光分别求位姿，M 默认不参加。

## 当前入口与结果身份

| 用途 | 当前位置 | 身份 |
| --- | --- | --- |
| 输入选择与原始数据 | [输入审计](basketball_input_audit/analysis.md)、`../input/篮球场/`、`../work/basketball/manifest.json` | 原始图像及选择协议；保留原件 |
| 全场视觉 SfM 阶段 | [全场报告](basketball_global_sfm.md)、`../work/basketball/global_sfm/stages/` | 历史阶段及共享内参模型，后续 BA 的起点 |
| 当前默认相机解 | [联合 BA 报告](basketball_ground_joint_ba.md)、`../work/basketball/global_sfm/ground_joint_ba/equal_control/sparse/0/` | 等权续算模型；后续诊断按此固定路径读取；尚非最终验收解 |
| 未采用的位姿候选 | `../work/basketball/global_sfm/ground_joint_ba/near_floor_2p0/sparse/0/` | 近地点加权对照，未替换默认模型 |
| 当前 LFS 输入 | [LFS A/B 报告](basketball_lfs_ab.md)、`../work/basketball/global_sfm/lfs_ab/equal_control/` | 原生鱼眼 COLMAP 数据集，536 张图全部训练 |
| 人工审阅记录 | [静态点核查](basketball_static_validation.md)、`../work/basketball/global_sfm/static_validation/basketball-static-annotations.json` | 用户提供的粗标原件；不能当精确像素真值，必须保留 |

`lfs_ab/near_floor_2p0/` 和 `lfs_ab/runs/` 保存对照输入及两次完整训练。等权只是目前保留的基线，近地点加权收益未达到晋升标准；两组训练视图结果均不是新视角验收。

## 报告导航

先读[输入审计](basketball_input_audit/analysis.md)、[全场视觉 SfM](basketball_global_sfm.md)、[静态点标注语义](basketball_static_validation.md)、[等权与近地点联合 BA](basketball_ground_joint_ba.md)、[LFS 训练 A/B](basketball_lfs_ab.md)，即可了解输入边界、当前模型和已验证的 3DGS 训练视图表现。其余报告按问题归档在此索引，均为历史证据，不意味着对应候选已写入默认模型。

| 主题 | 报告 |
| --- | --- |
| 初期前端与局部重建 | [LoMa 首轮诊断](basketball_loma_diagnostic.md)、[局部 SfM](basketball_local_sfm.md)、[第三片段迁移](basketball_transfer_diagnostic.md)、[轨迹一致性](basketball_track_consistency.md) |
| 全场模型和训练对照 | [全场 SfM](basketball_global_sfm.md)、[近地面位姿试验](basketball_ground_pose_probe.md)、[联合 BA](basketball_ground_joint_ba.md)、[近地面图像代理检查](basketball_ground_photo_proxy.md)、[LFS A/B](basketball_lfs_ab.md)、[轨迹清理候选](basketball_sfm_track_cleanup.md) |
| 匹配与有效域排查 | [前端证据](basketball_sfm_frontend_evidence.md)、[前端后续试验](basketball_sfm_frontend_followup.md)、[鱼眼 FOV/场地候选](basketball_sfm_fov_court_probe.md) |
| 球场地面匹配 | [三帧光流](basketball_sfm_court_flow_probe.md)、[定向 LoMa 与平面运动](basketball_sfm_court_plane_probe.md)、[轨迹平面核查](basketball_sfm_court_trajectory_probe.md)、[密集配准](basketball_sfm_court_dense_probe.md) |
| 场地角点和静态对照 | [场地结构角点](basketball_sfm_court_anchor_probe.md)、[全场双角扫描](basketball_sfm_court_paint_pair_probe.md)、[白线交会](basketball_sfm_court_white_line_probe.md)、[R 帧近远静态结构](basketball_sfm_r_static_depth_probe.md)、[L150 近地面轨迹](basketball_sfm_l150_depth_probe.md)、[L154 白斑](basketball_sfm_l154_patch_control.md)、[L154 右角和位姿假设](basketball_sfm_l154_right_corner_probe.md) |

## 当前结构和占用

| 目录 | 约占用 | 作用与整理判断 |
| --- | ---: | --- |
| `input/` | 5.8 GB | 原始图像、pose、`.pro`、LAS；即使先验不可信，也必须保留原件 |
| `work/` | 9.1 GB | 中间结果、冻结协议、审阅图、模型和 LFS 训练；不能把整个目录当缓存 |
| `work/basketball/global_sfm/lfs_ab/` | 4.5 GB | 两组训练快照与渲染；其中两个 `project.licht` 各约 1.5 GB，图像对比产物也占较多空间 |
| `work/basketball/features/` | 1.9 GB | 特征缓存；主缓存目录有 642 个 `.npz`，全场特征目录为 536 张图记录了绝对路径，迁移需改路径或重建目录 |
| `work/basketball/global_sfm/stages/` | 559 MB | 三个全场 SfM 阶段，后续实验的起点；保留 |
| `work/basketball/local_sfm/` | 530 MB | 初期局部重建与掩码；其中旧模型下载缓存约 210 MB，和复算成本有关 |
| `.venv/`、`.cache/torch/` | 7.0 GB、1.9 GB | 环境和模型权重下载缓存；理论上可重建，但重装和下载成本高 |
| `src/geosunpre/` | 2.3 MB | 56 个平铺模块、约 15,000 行；其中 23 个文件名含 `probe` |
| `reports/` | 1.7 MB | 本索引以外的 26 份 Markdown 报告及输入审计附件 |

仓库合计约 24 GB。`cli.py` 有约 609 行、60 个子命令，主流程和一次性诊断混排；根目录 `README.md` 约 221 行，现状更像按时间追加的实验日志。`global_sfm/` 同时存放主模型、A/B、众多 `sfm_*probe`、`_v2` 及 `_preimagehash` 目录，名字相似但不代表内容可互换。

当前 Git 仓库尚无提交，源码、报告和配置均未纳入提交历史；`input/`、`work/`、`.cache/`、`.venv/` 被 `.gitignore` 排除。截至本次盘点，29 份 Markdown 中的 371 个本地链接全部存在，其中 273 个指向 `work/`；部分脚本也写死了默认模型路径，特征目录还记录绝对路径。直接移动目录会同时破坏报告、脚本或复算入口。LFS 导出目录中的 `images/`、`masks/` 和共同点云使用符号链接，复制时应保留链接关系或明确展开。

## 清理顺序

1. **先固定基线。** 为源码、配置和报告建立 Git 提交；为输入选择、用户标注、默认 COLMAP 模型、LFS 导出和对照结果建立带大小与哈希的清单，并核实有独立备份。当前 Git 历史不能承担恢复责任。
2. **整理入口。** 保留现有命令和路径，先让根 README 聚焦安装、当前模型及复算入口；历史实验从本索引查找。随后再把 CLI 主流程和诊断命令拆开，保持旧命令可用。
3. **归档历史试验。** 逐项依据报告引用、源码读取、冻结协议及 `DIAGNOSTIC_ONLY.json` / `NOT_PROMOTED.json` 判定。优先将不再活动的完整试验打包到有清单的归档位置，而不是直接删掉 `_v2`、旧 mask 或 probe 目录；每次移动都要同步更新链接和路径。
4. **最后回收空间。** `.venv/`、下载权重及旧模型缓存可以在重建方法和网络可用性确认后回收。两个 LFS `project.licht` 和评价图虽大，但分别承载训练状态与已报告的图像对比；在有可验证归档前不列为删除对象。

本轮只建立索引和依赖边界，未移动或删除输入、模型、人工标注、训练结果及缓存，也未改变 SfM/LFS 行为。
