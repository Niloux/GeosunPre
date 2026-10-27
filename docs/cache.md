# 原生缓存

工作目录（`scene.json`、`state.json`、`result.json`、`cache/`）只在调试模式下长期保留。默认生产模式把工作目录放在交付目录同级的 `OUT.work`（或 `--work`），交付发布并回读校验后删除；未完成或失败时保留以便续跑。`state.json` 的 `debug` 记录模式，`output` 记录交付目录，续跑 `run --work WORK` 沿用该目录；旧版工作目录没有这两个字段，按调试模式处理。独立 `export`、`refine`、`cleanup` 需要调试模式的工作目录。

LAS 深度和法向监督默认随最终模型导出，`--no-las-supervision` 可关闭；这些导出选项不改变重建缓存或 pipeline_version。已有最终模型可独立 `export` 到新交付目录；监督算法、开关、参数或排除掩码改变时不得覆盖旧包。当前监督算法版本为 8，遮挡余量改为背景侧 2 像素、前景侧按 1 cm 换算到该处深度的像素，`--supervision-boundary` 只作用于语义掩码边界；版本 7 静态掩码缩放到监督图时改为对无效像素做面积缩放并要求结果严格为 0，消除浮点面积权重在非整数缩放的固定列上误判无效的竖线；版本 6 排除点间距超过近邻中位数 2 倍的零散回波，足迹系数改为 2；版本 5 法向改在 LAS 点云上以 kNN 间距与修剪 PCA 估计，深度和法向由切平面面元圆盘两遍混合投影得到；版本 4 投影后先剔除孤立异常点，遮挡断层改为两侧有支撑的台阶；版本 3 法向改用三维邻域稳健拟合，版本 2 分开深度与法向有效性并允许法向拟合邻域缺测。版本 1–7 的旧交付可继续 `check`，重新导出使用新目录。旧的无监督交付可用 `--no-las-supervision` 复用，补监督须指定新输出目录。监督请求身份包括原始 LAS、静态掩码及额外几何掩码的文件元数据，监督图与其有效数量在交付回读时校验。

所有数值小端。字符串为 uint64 字节长度及 UTF-8 字节，无终止符；格式标记也是这种字符串。不同版本不做自动兼容。

scene.json 保存曝光、初始相机和输入元数据。state.json 版本 1，保存冻结场景、配置、模型元数据和阶段产物元数据。文件变化时拒绝复用；输入、配置或模型改变时使用新 work。

state.json 的 pipeline_version=20 标识当前实现，恢复时须与程序的实现标记一致；不匹配时使用新目录。旧工作目录仍可独立 `export`、`refine` 或 `cleanup`，原结果不迁移、不改写。状态与结果保存本次执行的构建版本、时间和 EXE 路径。cache/sfm.json 保存几何验证、轨迹补全和各次 BA 的收敛、有效观测及耗时，`geometry_threads` 为几何验证使用的线程数（不影响验证结果），最终 result.json 同时包含此报告。unassigned_completion 记录最终标定后固定相机、对匹配连通分量内未分配观测重新三角化的新增点数和观测数；不重复占用已有特征。timing 细分缓存读取、轨迹构建、候选选择、位姿注册、三角化、过滤、检查点和着色耗时。导航误差按最终所选模型计算。

配置中的 `ba_workers` 默认 8，控制 SfM 全部 BA（增量注册与注册结束后的标定、全局精调）、LAS 联合优化和流水线原图精修的 BA 线程数（版本 20 起增量注册 BA 也用它，此前为 `workers`）；其他并行工作沿用 `workers`（默认 4）。每次 BA 保留既有 `build_seconds`、`linear_solver_seconds`、`jacobian_seconds`，新增 `timing` 中的观测建模、参数与先验建模、Ceres 预处理、迭代、后处理及纯残差计算耗时。`threads`、`preconditioner`、`termination`、评估次数、成功与失败步数、线性迭代总数、单轮最大值及触及迭代上限次数说明实际求解过程；`iteration_profile` 逐轮记录代价、梯度、信赖域、步求解耗时和接受状态。逐轮耗时与汇总耗时是同一过程的不同层级，不能将各层统计重复相加。线程数改变使用新工作目录；算法精度及质量门限不因线程数改变而放宽。迭代上限超过 100 或带 LAS 平面、导航约束的 BA 以相对代价下降 1e-6 判定收敛（版本 18 起，此前为 1e-7），其余 BA 为 1e-5；版本 17 的末段迭代每轮代价下降不足百万分之一，相机中心变化在微米级。

cache/{masks,features,matches}_profile.json 保存本次阶段的计算及复用数量、读图或缓存读取、预处理、输入等待、写入和推理耗时，最终 result.json 的 frontend 字段包含这些报告。匹配报告还保存配对总数、每图连接数及定向补边的图像编号。GPU 推理由 CUDA 事件测量；累计工作时间可以重叠，不能简单相加为总耗时。恢复运行的报告仅计本次计算，补匹配记录在 matches 报告的 repairs 数组，续跑沿用已完成的轮数。

`cache/features/索引.bin`：字符串标记 `GEOSUN_FEATURES_2`；uint32 image、dimension；uint64 N（32–3072）；N 条 float32 `(x,y,nx,ny,score)`；N×256 个 float32 描述子；256 个 float32 检索向量（描述子均值的 L2 归一化）；N 条 uint8 RGB。颜色取原图对应像素，避免 SfM 末尾重复解码照片。

`cache/matches/a_b.bin`：标记 `GEOSUN_MATCHES_1`；uint32 a、b（a<b）；uint64 M；M 条 `(uint32 feature_a,feature_b,float32 score)`。候选在 pairs.json。每幅图另外选择导航位置 5 m 内、曝光时间相隔超过 10 s 的最近 20 幅图作为回访候选，去掉其中同镜头且导航速度夹角超过 120°（两者都超过 0.1 m/s）的图：反向经过时车体转了半圈，同一镜头朝向对侧，版本 18 的生产车间中这类候选约一半与本图没有共同重建点；L/R 不作同步假设，候选仍须通过描述子匹配和鱼眼几何验证。匹配报告的 navigation_added_pairs 记录本次新增的导航候选数。初始近邻按曝光排序编号保留间隔 1–4、6、8、10、12；检索按编号距离 `[12,48)`、`[48,160)`、`[160,N)` 分段，每段从前 4 个候选选最多 3 个，排除同镜头相隔不足 2 秒的重复候选，L/R 不作同步假设。补匹配由 `sfm_repair_rounds` 控制，取值 `0`（默认）、`1` 或 `2`；启用时保持已有边和特征身份，针对未注册或未达到每图 32 个观测、8 个覆盖格子的图像扩展密集近邻及检索，补边不执行初始稀疏化及检索去重。

`cache/{visual,aligned}.bin`：标记 `GEOSUN_RECONSTRUCTION_1`；相机 JSON 字符串；uint32 gauge_image、scale_image；float64 gauge_baseline；uint64 图像数、点数。图像为 uint32 camera、uint8 registered、7 个 float64 `qx qy qz qw tx ty tz`、uint64 N、N 对 float64 像素坐标。点为 3 个 float64 XYZ、3 个 uint8 RGB、float64 error、uint64 轨迹长度及 `(uint32 image,feature)` 列表。点引用由轨迹重建；空轨迹保留内部编号，导出时省略。

掩码为 `cache/masks/索引.png`。阶段完成后才提交状态，中断时保留完成的单图/单对缓存；`.writing` 不作为完成依据。残留交付 `.pending` 在重试时改名保留，再重新生成。

`cache/incremental.bin` 使用同一重建格式，保存 SfM 的部分注册结果，每次局部优化后原子更新。补匹配及中断恢复会重新验证匹配几何、重建候选关联，保留已有位姿、三维点和观测身份；该文件不作为视觉验收或导出依据。

默认首次停滞就对已注册部分完成最终优化和验收；显式启用补匹配时，在轮数用完后处理已注册部分。通过后写入 visual.bin 并继续配准、可选精修、独立清理和导出。所有重建缓存保留全部输入索引及 registered 标记，未注册图像不出现在点轨迹中。cache/unregistered.json 保留 SfM 阶段未注册的原输入索引；全部注册后删除。state.registration 保存最终所选模型的注册统计，sfm 报告的 quality.registration 保存后处理前的统计。status=completed 表示交付完成，是否全注册看最终 registration.complete。

默认阶段为 `prepare → masks → features → matches → sfm → align → shutter → cleanup → export`。配置 `refine` 默认为 false，首次运行 `--refine` 等价于设为 true，选择随配置冻结；启用后在 align 与 shutter 之间增加 refine 阶段。align 产物为 `cache/aligned.bin` 和 `cache/alignment.json`，后续处理不改写它们。

启用 refine 时，写入 `cache/refinement_candidate.bin`、`cache/refined.bin`、`cache/refinement.json` 和中间结果 `cache/refinement_result.json`。refined.bin 保存相机精修后选中的模型，候选未通过时保存对齐模型；该阶段不执行强力清理、不写最终 result.json。关闭时不生成这些缓存，最终报告记录 refinement.enabled=false。

配置 `rolling_shutter` 默认为 true，在 align（或已启用的 refine）与 cleanup 之间执行 shutter 阶段，写入 `cache/shutter_candidate.bin`、`cache/shutter.bin`、`cache/shutter.json` 和中间结果 `cache/shutter_result.json`。shutter_candidate.bin 是换算到参考位姿的候选，shutter.bin 保存所选模型：通过验收时为候选，否则为上一步模型。shutter.json 在报告之外保存 `motion`：`readout_axis`（rows 或 columns）、读出比例约定、相机点公式、两镜头 `readout_s`，以及每幅已注册照片的 `rotation_rad`（参考相机系、整帧读出）和 `velocity_m_s`（参考相机系）。旧版工作目录中的 `cache/shutter/`（距离网格与校正后掩码）不再生成或读取。配置为 false 时不生成这些缓存，最终报告记录 shutter.enabled=false。

cleanup 阶段默认对上一步所选模型独立清理与验收，写入 `cache/cleanup_candidate.bin`、`cache/cleaned.bin`、`cache/cleanup.json` 和最终 `result.json`。可靠 LAS 平面支撑少于 200 条时不采用清理；清理候选未通过时同样保留清理前模型，cleaned.bin 保存所选结果，cleanup.used=false 及 reason 记录原因。清理只处理稀疏点及轨迹，相机内参和位姿保持不变。最终报告分别保存 refinement 和 cleanup；cleanup.source_model 记录输入缓存为 aligned、refined 或 shutter，quality 与导航误差按最终所选模型计算。

`selected_model=cleaned` 指定读取 cleaned.bin，`shutter` 指定 shutter.bin，`refined` 指定 refined.bin，`aligned` 或未声明字段时读取 aligned.bin；旧结果的选择不变。result.json 的 shutter.used=true 表示所选模型已换算到参考位姿，导出交付原图与 `rolling_shutter.json`。独立 `refine` 只执行相机精修、选择 refined.bin，来源已卷帘校正时拒绝执行；独立 `cleanup` 只清理稀疏结构、选择 cleaned.bin，来源已卷帘校正时复制 shutter.json；均读取来源的所选模型并写入新目录，保留来源目录。

各阶段完成后才提交状态，产物纳入文件元数据校验；中断时重跑该阶段，不将未完成候选视为成果。清理不会改写精修的中间产物，续跑可分别复用。新流程在 cleanup 完成后才允许导出；默认导出和独立 `export` 都使用 result.json 指定的模型，其元数据也用于交付包的请求身份。

精修误差统计的 `per_image` 按内部图像编号列出样本数、中位数和 P90，`cells` 再按原图 16×12 网格统计，格子编号为 `row*16+column`（从 0 开始）。`image_names` 将内部编号映射到图像文件名；它不等同于导出 COLMAP 的 image_id。`validation_coverage.images_without_holdout` 明确列出没有留出观测的已注册图像。全局验收通过不表示这些照片或未覆盖区域已通过独立校验；此报告中的留出误差是重三角化后的重投影误差，不能代替指定表面的原图投影叠加检查。

`pre_calibration_completion` 记录每次末尾标定支撑检查前的轨迹补全，新增点参与随后的 BA。标定两档的 `calibration_selection` 均记录可靠轨迹的逐图观测数、8×8 覆盖格子和角度带计数。支撑不足的检查也保留报告：`used=false`、`insufficient_images` 列出对应曝光；可用子集收尾排除这些曝光后重新检查，最终 `unregistered.json` 和交付 registration 包含全部省略曝光。

原图精修 `refinement_version=7` 的 `validation_coverage` 记录逐图训练及留出覆盖、限值和不足名单；覆盖不足时不运行相机优化。精修报告不再内嵌 structure_filter，清理报告位于独立的 cleanup。`optimization_reference` 说明增量优化的原投影与法向位移基准，`optimization.reference_observations` 记录使用原投影的训练观测数；优化代价包含这些先验，不能直接作为实际观测误差。

`training_error_px`、`validation_before_px/after_px` 均使用实际像素；后者包含全部留出轨迹，`photo_validation_before_px/after_px` 单独统计成功光度精修的留出轨迹，用于改善门限。逐图及有至少 8 个样本的格子退化名单另存样本数和前后中位数、P90。

清理的 `las_support_passes` 记录抽样元数据、完整 LAS 复核数、`rescued_from_sample_gaps` 和最终无支撑数，以及抽样、抽样检索和完整复核的耗时。完整复核把待查点按支撑距离宽的网格登记到周围 27 格，逐条 LAS 记录只查一格。`nearest_las_m` 是至已搜索到的有效 LAS 点的距离，为完整点云最近距离的上界；仅在距离大于门限时需要遍历完整 LAS 确认无支撑。临时点云与原始 LAS 块保持有界内存。
