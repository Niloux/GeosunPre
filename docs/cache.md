# 原生缓存

工作目录（`scene.json`、`state.json`、`result.json`、`cache/`）只在调试模式下长期保留。默认生产模式把工作目录放在交付目录同级的 `OUT.work`（或 `--work`），交付发布并回读校验后删除；未完成或失败时保留以便续跑。`state.json` 的 `debug` 记录模式，`output` 记录交付目录，续跑 `run --work WORK` 沿用该目录。独立 `export` 需要调试模式的工作目录。

深度和法向监督默认随最终模型导出，默认来源为解耦混合监督（`--supervision-source moge`，`las_cleaned_moge2_normals`），`--supervision-source las` 选用纯 LAS 投影监督，`--no-las-supervision` 关闭全部监督；这些导出选项不改变重建缓存。已有最终模型可独立 `export` 到新交付目录；监督算法、开关、参数或排除掩码改变时不得覆盖已有交付包。

解耦混合监督将 LAS 投影深度经 MoGe-2 边缘屏蔽带与多面逆深度 RANSAC 清洗后输出（对法向一致且深度比不超过 3× 的实体平面免于穿透误删），并在非边缘实体平面上优先写入 LAS 面元法向、其余空洞与非平面区域由 MoGe-2 密集法向回填。几何引擎身份属于监督请求，见[输出约定](output.md)。遮挡余量为背景侧 2 像素、前景侧按 1 cm 换算到该处深度的像素，`--supervision-boundary` 只作用于语义掩码边界。静态掩码缩放到监督图时对无效像素做面积缩放并要求结果严格为 0。LAS 监督排除点间距超过近邻中位数 2 倍的零散回波，法向在 LAS 点云上以 kNN 间距与修剪 PCA 估计，深度和法向由切平面面元圆盘两遍混合投影得到，投影后先剔除孤立异常点，遮挡断层取两侧有支撑的台阶。监督请求身份包括原始 LAS、静态掩码及额外几何掩码的文件元数据，监督图与其有效数量在交付回读时校验。

所有数值小端。字符串为 uint64 字节长度及 UTF-8 字节，无终止符；格式标记也是这种字符串。缓存格式不做自动兼容，格式或实现改变后使用新工作目录。

scene.json 保存曝光、初始相机、调用方给出的全部输入路径（`inputs`：`project`、`photos`、`camera_positions`、`navigation`、`enh`、`las`）及其文件元数据。state.json 保存冻结场景、配置、模型元数据和阶段产物元数据。续跑和独立导出沿用已记录的输入路径。切换解算项目，或输入、配置、模型或程序实现改变时，使用新工作目录。状态与结果保存本次执行的构建版本、时间和 EXE 路径。

SfM 在导航坐标系中注册并带导航先验，同步 L/R 帧在卷帘 BA 中带刚性 rig 软约束。几何验证的五点假设由内置 Eigen 求解器生成：零空间参数化后以十个三次约束消元得到乘 x 的作用矩阵，取实特征向量。cache/sfm.json 保存几何验证、轨迹补全和各次 BA 的收敛、有效观测及耗时，`geometry_threads` 为几何验证使用的线程数（不影响验证结果），最终 result.json 同时包含此报告。unassigned_completion 记录最终标定后固定相机、对匹配连通分量内未分配观测重新三角化的新增点数和观测数；不重复占用已有特征。timing 细分缓存读取、轨迹构建、候选选择、位姿注册、三角化、过滤、检查点和着色耗时。导航误差按最终所选模型计算。

配置中的 `ba_workers` 默认 8，控制 SfM 全部 BA（增量注册与注册结束后的标定、全局精调）、LAS 联合优化的 BA 线程数；其他并行工作沿用 `workers`（默认 4）。每次 BA 记录 `build_seconds`、`linear_solver_seconds`、`jacobian_seconds`，`timing` 给出观测建模、参数与先验建模、Ceres 预处理、迭代、后处理及纯残差计算耗时。`threads`、`preconditioner`、`termination`、评估次数、成功与失败步数、线性迭代总数、单轮最大值及触及迭代上限次数说明实际求解过程；`iteration_profile` 逐轮记录代价、梯度、信赖域、步求解耗时和接受状态。逐轮耗时与汇总耗时是同一过程的不同层级，不能将各层统计重复相加。线程数改变使用新工作目录；算法精度及质量门限不因线程数改变而放宽。迭代上限超过 100 或带 LAS 平面约束的 BA 以相对代价下降 1e-6 判定收敛，其余 BA 为 1e-5。

cache/{masks,features,matches}_profile.json 保存本次阶段的计算及复用数量、读图或缓存读取、预处理、输入等待、写入和推理耗时，最终 result.json 的 frontend 字段包含这些报告。匹配报告还保存配对总数、每图连接数及定向补边的图像编号。GPU 推理由 CUDA 事件测量；累计工作时间可以重叠，不能简单相加为总耗时。恢复运行的报告仅计本次计算，补匹配记录在 matches 报告的 repairs 数组，续跑沿用已完成的轮数。

`cache/features/索引.bin`：字符串标记 `GEOSUN_FEATURES`；uint32 image、dimension；uint64 N（32–3072）；N 条 float32 `(x,y,nx,ny,score)`；N×256 个 float32 描述子；256 个 float32 检索向量（描述子均值的 L2 归一化）；N 条 uint8 RGB。颜色取原图对应像素，避免 SfM 末尾重复解码照片。

`cache/matches.bin`：单文件紧凑匹配归档，文件头 8 字节魔数 `GSMATCH1`，每条记录依次为 `uint32 a、b（a<b）、count`、`count` 条 `(uint32 feature_a, feature_b, float32 score)` 及末尾 `uint32` 完整标记 `0x4D454E44`（`MEND`），支持中断后自动截断末尾半截记录并按二维分块局部性续跑未完成像对。候选在 pairs.json。局部近邻按 L/R 各自曝光序建立：同镜头步长 `{1, 2, 3, 4, 5, 8}` 且导航位置相距不超过 15 m（短窗保留反向退步边）；异镜头取曝光时间差 `|Δt| <= 2.5 s`、15 m 内的最近至多 4 帧，不假设同步。导航回访候选取曝光时间相隔超过 10 s 的候选（同镜头 5 m 内排除反向运动，至多 5 个；异镜头 4 m 内至多 2 个，合计最多 6 个），同/异镜头池内先按相隔 `>= 12 s` 轨迹通次各取最近 1 帧，再按相隔 `>= 2.5 s` 且距已选 ≥ 1 m 补足，避免单一通次占满配额；反向运动定义为同镜头且导航速度夹角超过 120°、两者速率均超过 0.1 m/s。匹配报告的 `navigation_added_pairs` 记录本次新增的导航候选数。初始外观检索在局部窗外、导航位置不超过 20 m 且 `|Δt| > 10 s` 时过滤反向运动的候选中分 `|Δt| > 30 s` 与 `<= 30 s` 两段排序，各取前 8 按同镜头 `< 2 s` 去重（先取至多 2 个远距回环、再从中距补足至多 3 个，不足则由远距补齐）。补匹配由 `sfm_repair_rounds` 控制，取值 `0`（默认）、`1` 或 `2`；启用时保持已有边和特征身份，针对未注册或未达到每图 32 个观测、8 个覆盖格子的图像在同镜头 `±24` 步、异镜头 `|Δt| <= 12 s`（至多 12 个）及 `[6,24)`、`[24,80)`、`[80,N)` 三段检索（每段前 12 个）扩展密集边。

`cache/{visual,aligned}.bin`：标记 `GEOSUN_RECONSTRUCTION`；相机 JSON 字符串；uint32 gauge_image、scale_image；float64 gauge_baseline；uint64 图像数、点数。图像为 uint32 camera、uint8 registered、7 个 float64 `qx qy qz qw tx ty tz`、uint64 N、N 对 float64 像素坐标。点为 3 个 float64 XYZ、3 个 uint8 RGB、float64 error、uint64 轨迹长度及 `(uint32 image,feature)` 列表。点引用由轨迹重建；空轨迹保留内部编号，导出时省略。

掩码为 `cache/masks/索引.png`。导出时持续人物及经 LAS 核验的天空误检改回静态后的掩码为 `cache/static_masks/索引.png`，其身份（LAS、所选模型、卷帘运动、来源掩码及判定规则）和报告在 `cache/static_masks.json`（交付 `manifest.json` 同时保存不含 `component_rows` 的摘要统计，便于生产模式删除工作目录后回溯）；身份相同时复用，否则重算并替换，不改动 `cache/masks`。阶段完成后才提交状态，中断时保留完成的单图/单对缓存；`.writing` 不作为完成依据。残留交付 `.pending` 在重试时改名保留，再重新生成。

`cache/incremental.bin` 使用同一重建格式，保存 SfM 的部分注册结果，按时间间隔（距上次保存不少于 20 s 且每注册 8 幅图像触发一次）及停滞或标定剔除时原子更新。补匹配及中断恢复会重新验证匹配几何、重建候选关联，保留已有位姿、三维点和观测身份；该文件不作为视觉验收或导出依据。

默认首次停滞就对已注册部分完成最终优化和验收；显式启用补匹配时，在轮数用完后处理已注册部分。通过后写入 visual.bin 并继续配准、独立清理和导出。所有重建缓存保留全部输入索引及 registered 标记，未注册图像不出现在点轨迹中。cache/unregistered.json 保留 SfM 阶段未注册的原输入索引；全部注册后删除。state.registration 保存最终所选模型的注册统计，sfm 报告的 quality.registration 保存后处理前的统计。status=completed 表示交付完成，是否全注册看最终 registration.complete。

默认阶段为 `prepare → masks → features → matches → sfm → align → shutter → cleanup → export`。align 产物为 `cache/aligned.bin` 和 `cache/alignment.json`，后续处理不改写它们。

配置 `rolling_shutter` 默认为 true，在 align 与 cleanup 之间执行 shutter 阶段，写入 `cache/shutter.bin`、`cache/shutter.json` 和中间结果 `cache/shutter_result.json`。shutter.bin 保存所选模型：通过卷帘验收时为换算到参考位姿的候选；卷帘未通过但全局快门 rig 回退（`global_rig_fallback.used=true`）通过验收时为 rig 修正后的全局快门模型（`shutter.used=false`，不交付 `rolling_shutter.json`）；否则为上一步模型。shutter.json 在报告之外保存 `motion`：`readout_axis`（rows 或 columns）、读出比例约定、相机点公式、两镜头 `readout_s`，以及每幅已注册照片的 `rotation_rad`（参考相机系、整帧读出）和 `velocity_m_s`（参考相机系）；报告另含两轴轻量预筛 `axis_screen` 与 `axis_attempts`（`evaluated: "full" | "screen"`）。配置为 false 时不生成这些缓存，最终报告记录 shutter.enabled=false。

cleanup 阶段默认对上一步所选模型独立清理与验收，写入 `cache/cleaned.bin`、`cache/cleanup.json` 和最终 `result.json`。可靠 LAS 平面支撑少于 200 条时不采用清理；清理候选未通过时同样保留清理前模型，cleaned.bin 保存所选结果，cleanup.used=false 及 reason 记录原因。清理只处理稀疏点及轨迹，相机内参和位姿保持不变。cleanup.source_model 记录输入缓存（aligned 或 shutter），quality 与导航误差按最终所选模型计算。

`selected_model=cleaned` 指定读取 cleaned.bin，`shutter` 指定 shutter.bin，`aligned` 或未声明字段时读取 aligned.bin。result.json 的 shutter.used=true 表示所选模型已换算到参考位姿，导出交付原图与 `rolling_shutter.json`。

各阶段完成后才提交状态，产物纳入文件元数据校验；中断时重跑该阶段，不将未完成候选视为成果。cleanup 完成后才允许导出；默认导出和独立 `export` 都使用 result.json 指定的模型，其元数据也用于交付包的请求身份。

`pre_calibration_completion` 记录每次末尾标定支撑检查前的轨迹补全，新增点参与随后的 BA。第一次检查后联合标定主点及全部内参；第二次检查后只在固定内参下精调相机和点，内参随后由卷帘阶段继续精调。标定两档的 `calibration_selection` 均记录可靠轨迹的逐图观测数、8×8 覆盖格子和角度带计数。支撑不足的检查也保留报告：`used=false`、`insufficient_images` 列出对应曝光；可用子集收尾排除这些曝光后重新检查，最终 `unregistered.json` 和交付 registration 包含全部省略曝光。

清理的 `las_support_passes` 记录抽样元数据、完整 LAS 复核数、`rescued_from_sample_gaps` 和最终无支撑数，以及抽样、抽样检索和完整复核的耗时。完整复核把待查点按支撑距离宽的网格登记到周围 27 格，逐条 LAS 记录只查一格。`nearest_las_m` 是至已搜索到的有效 LAS 点的距离，为完整点云最近距离的上界；仅在距离大于门限时需要遍历完整 LAS 确认无支撑。临时点云与原始 LAS 块保持有界内存。
