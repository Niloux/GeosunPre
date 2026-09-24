# GeosunPre

面向 LichtFeld-Studio 的异步双鱼眼 SfM 预处理。已完成数据准备、LoMa 前端诊断、局部 SfM/内参候选、第三片段迁移，以及全部 536 张入选 L/R 图像的视觉建图、固定内参 BA、闭合三图训练轨迹筛选和全场共享内参对照。等权与近地点加权相机解已导出 COLMAP 并完成 LFS 全训练 A/B；训练视角差异很小，目前继续优化 SfM 图像一致性。未进行 LAS 配准，也未认定最终相机解。

当前模型、报告导航和清理边界见[项目结构与实验索引](reports/README.md)。
工程主线与实验代码的目标边界见[工程组织方案](docs/engineering_structure.md)。

本项目当前约定：L/R 使用 `OPENCV_FISHEYE`，`.pro` 的 `p3` 对应第四个鱼眼畸变参数；每张图像有独立曝光位姿，L/R 分别共享内参。仅使用具有精确同名 CameraPos 记录的 L/R，M 默认排除。输入 pose 的数值不参与几何拟合。

## 环境与数据准备

使用仓库指定的 Python 3.13 和 uv。LoMa 固定在 `8fb59c458f2a2ef44314f462467b229c27786bf0`；完整依赖由 `uv.lock` 锁定。首次运行会下载 CUDA PyTorch 和模型权重，需要网络与 GPU 访问。

```bash
uv run geosunpre prepare \
  --input input/篮球场 \
  --work work/basketball \
  --principal-y up \
  --fov 160
```

产出 `manifest.json`、`images.txt` 和 `fov_masks/L.png`、`fov_masks/R.png`。篮球场应选中 536 张、第 33–300 帧，每张有不同的 `exposure_id`。文件名的帧编号仅作元数据，不构成同步 rig frame。

`--principal-y up` 明确采用标定主点 y 轴向上的初始约定，尚未认定该约定正确。内部使用 COLMAP 角点坐标：首个像素中心为 `(0.5, 0.5)`。160° 是明确记录的总视场候选，不是本项目测得的硬件视场。

可通过 `--mask-root <目录>` 提供额外的白色有效、黑色忽略的语义掩码，命名如 `L/0000/<原始文件名.JPG>.png`，尺寸必须与原图一致。未提供时只使用硬 FOV，不能宣称已排除天空和动态目标。原图不会被修改。

## 小范围 LoMa 诊断

```bash
uv run --extra loma geosunpre diagnose --work work/basketball
```

默认选择篮球场的 32 张代表图像、108 对时间邻接/跨相机/首尾回环候选。也可用 `--frames 199,200,201,202` 指定帧号；指定帧必须已经入选。`--detector-size`、`--keypoints`、`--match-threshold`、`--max-error` 控制提取分辨率、有效点数、匹配阈值和原图像素几何阈值。

每图只提取一次，保留稳定的关键点 ID，按有效域过滤后按网格分配点数。缓存包含图像/掩码内容哈希、标定和提取配置；不同图像对复用同一份特征。LoMa 权重缓存在项目 `.cache/torch/`。

几何诊断使用 pycolmap 4.2.0 的相机感知 `estimate_relative_pose`，只拟合射线空间的 E 与相对位姿。通用 `estimate_calibrated_two_view_geometry` 仍可能切换到原始像素 F，因此不用于本鱼眼流程。预留约 20% 原始匹配不参与拟合，再在原始像素单位下评估包含鱼眼反投影导数的极线距离。报告保留原始匹配、拟合内点、几何筛选结果和轨迹冲突计数，并标记低视差图像对。**极线距离、两视图三角化重投影误差和最终多视图重投影误差是不同指标。** 本命令只完成前两项与特征轨迹连接，没有多视图 BA。

结果目录由配置指纹命名，路径记录在 `work/basketball/latest_diagnostic.json`。其中：

- `index.html`：可离线浏览、按图像对类型筛选的匹配总览。
- `report.json`、`pairs.csv`、`pairs/*.json`：统计及逐对来源。
- `pairs/*.npz`：固定关键点 ID 的原始匹配、置信度、留出标记、相对位姿和残差。
- `tracks.json`：跨图特征轨迹，拒绝包含同一图像两个不同特征的冲突合并。
- `principal_conventions.json`：同一批匹配与相同留出划分下的主点符号对照。
- `prior_conventions.json`：四元数顺序/变换方向对照，pose 不参与拟合。
- `rotation_cycles.json`：连通分量、三图旋转闭合误差及每条边所在闭环的统计。异常闭环用于提示复核，不会自动删除某一条边。

“几何通过”只是诊断阈值下成立，不等于静态实体对应已人工确认，也不等于通过 3DGS 精度验收。主点和 pose 对照均不会自动覆盖输入。

## 连续片段的局部重建

```bash
uv sync --extra loma --extra segmentation --group dev
uv run --extra loma --extra segmentation geosunpre local-sfm --work work/basketball
```

也可通过 `--stage masks|frontend|solve` 分阶段运行。固定试验片段为 70–94、190–214，各 50 张 L/R。输出位于 `work/basketball/local_sfm/`，`index.html` 为对照总览，`mask_review.html` 为原图与掩码对照。

- 掩码使用固定版本 SegFormer，在五个虚拟针孔视图上预测天空和人物，映射回原图后结合人工检查的设备轮廓。虚拟视图不会成为 SfM 中的新增相机。硬 FOV 独立保留；植被仍由几何阶段筛选。第一版将球场误判为草地/游泳池并过度剔除，已标记废弃。
- LoMa 检测长边 1536、每图最多 3072 个有效点；两个片段各 357 个候选图像对。约 20% 的合格原始长轨迹在几何验证前整条留出，并移除训练图中所有接触留出观测的边。
- 数据库采用两个单相机 rig，每个 frame 恰好一个独立曝光；L/R 没有固定外参或同步约束。输入导航 pose 和 LAS 不写入数据库。
- 显式构造鱼眼 E 模型的初始图像对与三角化点，从已有种子继续增量注册，避免默认初始化重新选择普通 F。
- 比较固定内参、焦距与低阶畸变、全部径向畸变、再放开主点四阶段。每阶段共享同一组训练观测；L/R 各自共享内参。参数范围是探索性的数值限制，并非先验真值或实测协方差。
- 主点阶段增加正深度障碍项，并对所有训练观测检查正深度、对 160° 域检查投影单调性。求解器状态和代价下降不能替代这些检查；出现过的非物理解保存在标记废弃的目录中。
- 验证点由两个完全留出的支持观测三角化，用其余留出观测检验预测。全部原始候选轨迹与高置信闭合三图子集分别报告；子集的三个直接匹配置信度均 ≥0.9，选择不使用相机姿态或查询重投影误差。它们仍不是人工标注真值。

各片段的 `stages/<阶段>/sparse/0` 包含 COLMAP 二进制和文本模型；图像根目录为原始 `input/篮球场/Image`。这些局部模型用于对照试验，尚未认定为最终 LichtFeld-Studio 训练数据。输入、掩码、特征或留出划分改变后，不得直接复用旧模型；运行约定会检测数据变化。

## 原图复核与第三片段检查

```bash
uv run --extra loma --extra segmentation geosunpre review-residuals --work work/basketball
uv run --extra loma --extra segmentation geosunpre transfer-check --work work/basketball
uv run geosunpre refinement-probe --work work/basketball
uv run geosunpre principal-ablation --work work/basketball
```

`transfer-check` 也支持 `--stage masks|frontend|solve|all`。第三段固定为 145–169，共 50 张 L/R，与前两段不重叠。拟合前保存三组候选参数、特征配置及验证规则；用初始内参建立一份训练拓扑，然后在相同的三维轨迹和观测上比较固定候选内参。优化变量只有每张图的独立位姿和三维点。

`--continue-iteration-limit` 可对达到迭代上限的候选进行一次有记录的继续求解，旧模型保留。不能只凭重启后的收敛标签确认数值稳定；本次另对各试验模型做了更大步长的数值检查，结果单独保存。

- `work/basketball/residual_audit/index.html`：44 个诊断样例，两个支持视图和查询图的原图局部、预测标记与目视说明。样例刻意包含大误差和低误差对照，不代表数据集整体错误率。
- `work/basketball/transfer_sfm/index.html`：第三段三组固定内参的原始/高置信留出对照与 L/R 分区。
- `work/basketball/transfer_sfm/transfer_145_169/principal_ablation/index.html`：仅改 R 的 `cy`，以及撤回这项改动的解释性对照。
- `work/basketball/feature_refinement/normal_070_094/index.html`：第一段训练观测的原图角点精化。内参、点/观测拓扑、留出支持点和查询点全部固定。当前精化方案没有取得明确留出收益，`NOT_PROMOTED.json` 记录了未采用的判断，默认流程继续使用原始 LoMa 观测。

后两项是在看到主要结果后追加的诊断，不能作为预先锁定的最终验收。原图、输入 `.pro`、原始 LoMa 缓存和前两段模型均保留。

## 针对性验证

```bash
uv run --extra loma --group dev pytest -q tests
```

测试覆盖鱼眼周边与 COLMAP 投影一致性、带外点的留出验证、大转角跨鱼眼 E 模型、同图重复轨迹冲突、旋转闭环方向，以及缺 pose/M 排除和独立曝光约定。局部实验还覆盖整条轨迹留出、训练隔离、失败预测分母、受约束 BA、正深度退化和障碍项导数。

初始数据分析与输入清单见 [篮球场输入审计](reports/basketball_input_audit/analysis.md)。
已运行的 LoMa 结果、闭环问题与后续边界见 [篮球场首轮诊断](reports/basketball_loma_diagnostic.md)。
掩码修正、两个连续片段的实际重建与留出预测结果见 [篮球场局部 SfM 对照](reports/basketball_local_sfm.md)。
原图误差来源、第三片段迁移和精化试验见 [篮球场迁移诊断](reports/basketball_transfer_diagnostic.md)。
闭合三图训练轨迹筛选的三段实测结果见 [篮球场轨迹一致性检查](reports/basketball_track_consistency.md)。

## 全量 L/R 数据准备与 SfM

```bash
uv run --no-sync --extra loma --extra segmentation geosunpre global-data --work work/basketball
uv run --no-sync geosunpre global-sfm --work work/basketball
```

`global-data` 分为 `masks|features|matches` 三步，可用 `--stage` 分阶段运行；已运行的局部 150 张修正版 mask 和匹配缓存会复用。全部 536 张仍限于精确同名 CameraPos 记录的 L/R，导航 pose 数值与 LAS 不进入拟合。图像外观检索只用于提出可能的远距离图像对，不提供位姿或几何真值。

`global-sfm` 会先在几何拟合前冻结整条原始轨迹留出和高置信闭合三图查询，再以 70–94 的视觉内参候选建立全场图。映射后比较固定内参下的全部训练轨迹和闭合三图支持轨迹，分别重新优化每张曝光的位姿与三维点；最后在同一筛选拓扑上有界优化 L/R 各自共享的内参。它支持 `--stage mapping|mapping-eval|solve|compare|all`；`mapping-eval` 单独对已保存的增量建图模型运行同一留出评估，`compare` 计算固定内参与筛选后模型在相同合格支持查询上的配对误差。所有阶段保留原生鱼眼观测和独立曝光约定。

当前全场增量建图已注册 536/536 张；固定内参 BA 的高置信闭合三图留出误差中位数 / P90 为 2.09 / 7.28 原图像素，训练轨迹筛选后为 2.01 / 7.05，筛选后共享内参优化为 1.92 / 6.85。全部原始长轨迹误差仍大，不能直接按注册率或高置信收益认定可用于高一致性 3DGS。实际结果与原图外观检查见[篮球场全场 SfM 阶段报告](reports/basketball_global_sfm.md)和 `work/basketball/global_sfm/index.html`。

```bash
uv run --no-sync geosunpre global-review --work work/basketball --candidate triangle_supported
uv run --no-sync geosunpre photometric-review --work work/basketball
```

`global-review` 从冻结的高置信留出轨迹中按镜头和径向分区选取 18 个诊断样例，逐例显示两个支持视图和查询视图的原图局部、预测和无标记版本；样例不代表全场错误率。`photometric-review` 在原生 JPEG 上统计几何验证匹配的 5×5 RGB 差异，属于外观诊断，不等于线性辐射或 3DGS 渲染误差。

## 原图静态点盲审

```bash
uv run --no-sync geosunpre static-review --work work/basketball
```

从几何拟合前冻结的 LoMa 留出轨迹中，按 L/R、四段时间和原图屏幕位置确定性抽取 240 个三视图候选，不参考现有 SfM pose 或残差。页面导出的 `basketball-static-annotations.json` 有 238 条粗标签，其中 114 条标为 `same_static`。**用户后来说明这些不是严格逐点标注：叶片/植被难确认的点被标为“动态”，绝大多数坐标未作精细修正。**因此旧报告中的 1.79 / 6.93 原图像素只是相对于原始 LoMa 查询的探索性残差，不能当作相机精度验收或前端点位准确的证明。原始数字、抽样隔离与[标注语义修正](work/basketball/global_sfm/static_validation/annotation_semantics.json)见[篮球场静态点核查报告](reports/basketball_static_validation.md)。

## 近地面轨迹位姿对照

以下近地面试验中旧称“人工确认”的点后来被用户澄清为粗略标签，绝大多数像素位置未精调；相关数字只作探索性对照，不能当作精确位姿验收。见[标注语义修正](work/basketball/global_sfm/static_validation/annotation_semantics.json)。

```bash
uv run --no-sync geosunpre ground-pose-probe --work work/basketball
uv run --no-sync geosunpre ground-pose-analyze --work work/basketball
```

从共享内参视觉模型中拟合并在原图上检查球场附近的薄平面；只对重投影误差 ≤2 px、轨迹长度 ≥4 的训练点有限增权，并按训练观测距离区分近场。试验固定所有三维点和 L/R 鱼眼内参，只优化 536 张异步曝光图像各自的位姿；原始 pose、LAS 和 114 条人工静态留出点都不进入优化。`work/basketball/global_sfm/ground_pose_probe/` 保存冻结协议、原图投影预览、各权重候选与按近场分组的结果。近场训练残差下降，但留出的近场地面代理点没有明确改善，所以这些候选**尚未替代全场 SfM 模型，也不是完整加权 BA 或 3DGS 验收结果**。数值、方法和局限见[篮球场近地面位姿对照](reports/basketball_ground_pose_probe.md)。

## 全场近地面加权联合 BA

```bash
uv run --no-sync geosunpre ground-joint-ba --work work/basketball
uv run --no-sync geosunpre ground-joint-analyze --work work/basketball
```

在同一原生鱼眼 Ceres BA 中，对冻结的可靠近场/远场地面训练观测分别给 2 / 1.25 倍鲁棒代价，与从相同模型启动的等权续算对照；位姿、三维点及两镜头共享内参共同优化，仍保留正深度、鱼眼单调性、内参边界和独立曝光。增权候选已收敛，近场训练误差中位数 0.959→0.888 px，但 30 条近场地面代理人工点中位数 1.547→1.548 px、≤2 px 仍为 70%。因此结果标记为**未采用**，没有覆盖全场模型。完整结果、远点漂移及局限见[篮球场近地面联合 BA 报告](reports/basketball_ground_joint_ba.md)。

```bash
uv run --no-sync geosunpre ground-photo-proxy --work work/basketball
```

用同一批人工确认近场点在原生鱼眼 JPEG 上做 60 组局部地面切平面跨视图对齐，等权与增权模型的配对 ZNCC 差值中位数为 +0.0043，38 组变好、22 组变差，低分尾部变差。这是对 3DGS 相关原图一致性的**代理检查**，尚不能代替相同训练设置下的 LichtFeld-Studio 渲染 A/B；详见[篮球场近场原图对齐报告](reports/basketball_ground_photo_proxy.md)。

## LichtFeld-Studio COLMAP 3DGS A/B

```bash
PYTHONPATH=src .venv/bin/python -m geosunpre lfs-export-ab --work work/basketball
bash work/basketball/global_sfm/lfs_ab/run_ab.sh
PYTHONPATH=src .venv/bin/python -m geosunpre lfs-analyze-ab --work work/basketball
```

导出后的两套数据均是原生 `OPENCV_FISHEYE` 的 COLMAP `sparse/0`，共用 536 张原始 L/R 图像、合成天空/人物/设备与候选 FOV 有效域掩码，以及同一份等权视觉 SfM 筛选点云。M、无同名 pose 的静止图像、输入 pose 数值和 LAS 不进入训练。LFS 完整配置已移除旧 LiDAR 初始化，关闭去畸变；**全部 536 张训练，不划测试集**。脚本已顺序跑完等权和近地点加权各 30,000 步，并产出训练视图配对渲染指标。最终近地点 MAE 平均仅降低 0.0535 个 8-bit 色阶，全图 PSNR 19.9286→19.9214 dB，未显示足够明确的视觉收益，因此暂保留等权相机解为默认。具体导入核对、近地点/鱼眼外圈比较规则及局限见[篮球场 LFS A/B 报告](reports/basketball_lfs_ab.md)。

## 后续 SfM 训练轨迹清理对照

```bash
PYTHONPATH=src .venv/bin/python -m geosunpre sfm-track-cleanup --work work/basketball --stage audit
PYTHONPATH=src .venv/bin/python -m geosunpre sfm-track-cleanup --work work/basketball --stage all
```

这轮仅在视觉 SfM 中删除低视差、同图重复关联和较大训练残差的轨迹/观测，再做原生鱼眼联合 BA；未改 LFS。清理后仍有 536/536 张图，BA 正常收敛，但三图留出收益很小；粗标 `same_static` 查询的 P90 从 6.933 升至 7.183 原图像素，**不能据此判定真实位姿精度变差**。缺少可靠的像素级收益证据，因此新候选未替代等权模型。筛选数量和对照见[篮球场 SfM 轨迹清理报告](reports/basketball_sfm_track_cleanup.md)。

## SfM 前端匹配证据与残差结构

```bash
PYTHONPATH=src .venv/bin/python -m geosunpre sfm-match-evidence --work work/basketball
PYTHONPATH=src .venv/bin/python -m geosunpre sfm-residual-structure --work work/basketball
```

这两项只读诊断使用原 LoMa 缓存、粗标查询与保存的等权视觉模型。高置信三视图中存在目视可见的错位，以及叶片/植被上难以确认的点；粗标签上的分数区分度低，**并不证明 LoMa 前端已足够准确**。同图留出查询的二维残差偏移也不能用来排除位姿问题。方法、逐点结果和新的标注语义见[篮球场 SfM 前端证据报告](reports/basketball_sfm_frontend_evidence.md)。

## 前端点位、检测分辨率与虚拟视图小范围试验

```bash
PYTHONPATH=src .venv/bin/python -m geosunpre sfm-lk-probe --work work/basketball
PYTHONPATH=src .venv/bin/python -m geosunpre sfm-lk-refinement --work work/basketball --stage all
uv run --no-sync geosunpre sfm-resolution-probe --work work/basketball
uv run --no-sync geosunpre sfm-virtual-view-probe --work work/basketball
uv run --no-sync geosunpre sfm-virtual-view-audit --work work/basketball
uv run --no-sync geosunpre sfm-virtual-cycle-probe --work work/basketball
```

在选取的相邻图像对里，部分已通过两视图几何的高分 LoMa 对应与双向稳定的原图局部跟踪仍有 >2 px 差异；光流不是真值，且许多图像对跟踪失败。基于两个邻图共识的保守点位精化只改了局部 50 张图约 1.9% 的训练观测，BA 留出误差几乎不变，故未晋升。2048 分辨率检测在两对 L 图像的局部指标有所改善，在 R 图像上结果混合，尚无全场切换依据。

五个临时虚拟针孔视图中的 LoMa 特征已映回原生鱼眼坐标，并对四对相邻图像和四组三帧图像与原生前端对照。它增加了不少建筑区域三帧闭环，但 L120–122 的下半画面三视图几何合格轨迹只从 60 增至 61，L150–152 从 25 增至 39；尚不能证明球场视角姿态更准确，故[暂不替换全场前端](work/basketball/global_sfm/sfm_virtual_cycle_probe/NOT_PROMOTED.json)。输入内参和保存的视觉 pose 均作为候选或诊断参考，而非真值。全部协议、原始结果和预览见[篮球场 SfM 前端后续试验](reports/basketball_sfm_frontend_followup.md)。

对可见球场铺装继续核查后发现，旧 160° 鱼眼 FOV 与下方 20% 整带设备掩码是两层不同截断。六张 L 图像的隔离 179° 试验只在新环带放行球场颜色候选，并有限恢复底部铺装：L120–122 的球场区域三视图几何合格轨迹 52→55，L150–152 为 37→42，增益仍小，故[暂不晋升](work/basketball/global_sfm/sfm_fov179_court_frontend_probe_v2/NOT_PROMOTED.json)。扩大视野不会自动解决铺装重复纹理与点位精度，超出当前鱼眼前半球可逆范围的像素也不能直接加入。输入内参、160°/179° 边界均不是标定真值；详见[近场铺装有效域与前端试验](reports/basketball_sfm_fov_court_probe.md)。

对六张球场 L 图再独立提角点并做三帧双向光流后，原始方案只留下 30/51 条图像闭环轨迹；L150 图像下方的 18 条没有一条通过旧视觉解的三视图检查，原像素裁剪也显示重复格纹的点位歧义。预先平滑纹理将闭环数降至 11/27，仍没有合格近场点。两版[暂不晋升](work/basketball/global_sfm/sfm_court_flow_probe_v2/NOT_PROMOTED.json)；这不是对旧 pose 正确性的证明，而是说明简单局部光流不足以提供可放心加权的近地面约束。证据与复算入口见[近场角点与三帧光流诊断](reports/basketball_sfm_court_flow_probe.md)。

朝球场取单个 1280×1280 临时针孔视图后，LoMa 在两组 L 三帧中形成 162/74 条高分闭环，旧视觉解三视图检查合格 148/48；但图像下方且远离原 LoMa 特征的合格点仅 5/1，原像素裁剪仍难确认点身份，故[不作为高权重 SfM 轨迹](work/basketball/global_sfm/sfm_court_down_loma_probe/NOT_PROMOTED.json)。从这些匹配拟合的球场平面单应跨三帧闭环中位差为 1.08/0.78 个临时视图像素，色块边界能整体对齐；现有视觉 pose 加同一平面在这两个短片段内大体兼容，因此[平面结果仅用于诊断](work/basketball/global_sfm/sfm_court_plane_probe/DIAGNOSTIC_ONLY.json)。详见[定向 LoMa 与球场平面运动核查](reports/basketball_sfm_court_plane_probe.md)。

沿 536 张 L/R 有 pose 的图像扫描球场可见区后，按图像内容选出 L/R 共 46 组三帧、138 张图，复用已有 LoMa 匹配检查 694 对候选：113 对能拟合球场平面单应。101 对相距不超过两帧的边全部有保存的共同 3D 轨迹；它们的逐边最佳平面与图像单应差中位数之中位数为 0.69 个 640 像素临时视图像素。唯一 195.48 像素的远距离异常 L210–L289 的高分匹配只落在球场铺装，且没有共同 3D 轨迹，暂不能当作 pose 漂移。覆盖 L/R 的 25 个图像单应三角形进一步显示，L169–R249 的直接单应与两条中转路径不一致；L156、L230 附近也有受稀疏跨帧铺装点影响的闭环异常。结果[仅用于诊断](work/basketball/global_sfm/sfm_court_trajectory_probe/DIAGNOSTIC_ONLY.json)，当前尚未找到足以据此晋升新 pose 的局部区段；见[球场 L/R 轨迹平面核查](reports/basketball_sfm_court_trajectory_probe.md)。

对 110 条球场单应另做隔离的灰度密集配准后，24 个可比图像三角形中 17 个闭环有所改善，L169–R249、L156–158、L230–232 的大偏差明显收敛；但 R188–R189 的错误移动达到约 62 个视图像素，且相邻边整体的旧 pose 兼容性没有改善。因此[密集单应不晋升](work/basketball/global_sfm/sfm_court_dense_probe/DIAGNOSTIC_ONLY.json)为新的 SfM 轨迹或 pose；详见[球场平面密集配准试验](reports/basketball_sfm_court_dense_probe.md)。

继续核对红/绿/蓝场地交点后，L50–52、R223–225 的三图角点已有现成 3D 轨迹；L210–L289 虽可让相似色块的交点经单应几乎重合，原图纹理和结构身份仍不足以支持跨远距离连接。L233–235 原生图清楚的红色场地左角则未被旧特征覆盖；沿 L231–L242 追踪成 12 视图候选后，用 L233、L239 两张三角化预测其余 10 张，中位误差为 1.80 原图像素，且 12 张角点 16 像素内均无已有 3D 轨迹。这些角点在保存的掩码和候选 160° 鱼眼有效域内都有效，提示可从特征选择环节补充近地面约束；但当前视觉 pose 已能较好解释该短段，尚未执行 BA 或更改 LFS 输入。角点原图、留出与保守边界见[球场结构角点核查](reports/basketball_sfm_court_anchor_probe.md)。

把同一红色铺装区域的两个交点作为图像侧提案扫描全部 536 张 L/R 原图后，找到 14 段至少 5 帧的连续双角候选，12 段能在原图继续定位。L125–129、L172–178、L233–243、L252–259 的四条左角轨迹均未被旧 3D 点覆盖，当前视觉解的多帧留出误差中位约 1.8–2.9 原图像素；但 HSV 阈值扰动表明，只有 L233–243 左角在整段都保持不超过 2.2 像素的检测位移，其余几条仍需精定位。R267–281 的双角视觉连续性值得复核，但原图定位对阈值敏感，不能把旧视觉解下的大残差归因于 R pose。部分自动候选还会错认篮架立柱，均未晋升为 BA 约束。筛选、原图接触表与逐点结果见[全场场地双角轨迹核查](reports/basketball_sfm_court_paint_pair_probe.md)。

进一步在原生图像上按红/绿、红/蓝、绿/蓝邻接独立拟合白漆边线后，R267–281 左角在相同有效留出帧上的旧视觉模型误差中位从 5.72 降至 1.44 原图像素，说明原 HSV 点位是主要偏差来源之一；但 R281 新点越过当前候选 160° FOV。R 右角在浅交角与宽白漆处仍有定位歧义，固定支撑帧 R277 的阈值扰动可达 7.46 像素，不能用其较大两视图留出误差判定 pose 错误。旧模型中 R 双角附近也已有部分 3D 轨迹，而 L233–243 左角白线位置仍 11/11 帧无 16 像素内旧 3D 点。当前只生成可复核的诊断，未增加 BA 约束或更改 LFS 输入；详见[白线交会与旧 SfM 轨迹核查](reports/basketball_sfm_court_white_line_probe.md)。

为核对 R277–281 是否有能支持位姿修正的独立证据，先在原图审查 LoMa 三边闭环，再冻结未入旧 3D 模型的篮架、灯杆与建筑特征。R277 留出的四个主点误差为 0.82–5.64 原图像素，但方向不一致；R280 近处篮架两点为 1.23、3.29 像素，远处建筑两点为 1.49、2.32 像素，也没有共同的大位移。因此此前 R 右侧白线角点的约 7–9 像素差异不能直接解释为 R pose 错误，当前不执行该段 BA 或更改 LFS 输入。[近远静态结构及逐帧留出核查](reports/basketball_sfm_r_static_depth_probe.md)保留了选择协议、原图和数值。

L150–155 的原生 LoMa 缓存每图在下部画面仍有 829–977 个特征，但五组三帧下部严格闭环仅 6–25 条，四个连续四帧窗口的下部六边闭环均为零。L151–153 四个未建图固定结构对 L152 的留出误差为 0.41–1.81 原图像素、方向不一致；几条红色铺装候选虽有 2.87–6.30 像素偏差，原图无法确认精确同名点，不能据此修正位姿。当前仍保留等权模型，先解决近地面跨帧点位身份与轨迹持续性；详见[球场 L150 近地面轨迹及静态对照](reports/basketball_sfm_l150_depth_probe.md)。

对白色磨损短划做五帧局部 LoMa 采样后，虽然得到 9 条附近两跳闭环链，但原图只能把两条白斑边缘轨迹保留为候选，尚非精确地面控制点。以相同 L153/L155 支撑预测 L154，两条候选均有约 5.0 原图像素误差、主要沿 y 方向；两处不同位置的固定立柱脚只有 0.38、0.90 像素误差。白斑局部单应和原图相关性与 L154 LoMa 位置相差约 0–2 像素，因而值得继续查近距离位姿、内参与地面轨迹，但尚不能给候选高权重或改 BA。完整冻结点位、原图审阅和同支撑审计见[球场 L154 白斑与静态结构核查](reports/basketball_sfm_l154_patch_control.md)。

右侧绿/蓝场地白线角点在 L153–155 也可由原图中心线定位；其 L154 留出残差 y 为 −2.43 像素，阈值扰动下保持 −3.16 至 −2.00 像素，方向与白斑相反。但此角点对应旧模型已有的 23 视图 L/R 轨迹，并非新增独立控制。只用白斑 A 修 L154 的最小范数旋转会使近处立柱误差涨至约 4–5 像素；平移虽保住立柱，却使该旧角点的 L154 重投影误差从 2.81 涨至 5.16 像素。两个假设均未写回模型；见[球场 L154 右角及位姿假设核查](reports/basketball_sfm_l154_right_corner_probe.md)。
