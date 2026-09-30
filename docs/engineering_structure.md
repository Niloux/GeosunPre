# 工程结构与运行说明

## 模块职责

```text
src/geosunpre/
  cli.py                  # run / export / check；独立 align / inspect-alignment / refine-alignment / joint-alignment
  config.py               # 场景选择、匹配后端及冻结身份
  device.py               # 版本化 FOV、主点约定与设备遮挡配方
  artifacts.py            # 内容哈希、原子 JSON 写入、不可变协议
  scene.py                # Geosun 输入校验、L/R 图像选择
  camera.py               # 原生鱼眼相机与像素约定
  masks.py                # SegFormer、固定设备遮挡、预览
  features.py             # LoMa 与内容缓存
  matching.py             # FlashAttention、诊断 math 路径及 CUDA Graph
  parallel.py             # 有界、有序的 CPU 并行任务
  profiling.py            # RSS / CUDA 采样、嵌套计时与空闲内存回收
  colmap.py               # COLMAP 原子发布和独立校验
  alignment/              # LAS 配准、联合 BA 与独立诊断部件
    runner.py             # 坐标假设、输入身份、空间留出、候选导出
    geometry.py           # 鲁棒相似变换、表面细化与误差统计
    structural.py         # 多尺度平面、视觉精度代理、方向权重与自由度覆盖保护
    joint.py              # 固定内参的图像 / pose / LAS 对照、源身份、独立候选导出
    joint_solver.py       # 全局 Sim3、原生 Ceres 残差、图像及几何接受条件、回退
    joint_report.py       # 离线逐图像对照、LAS 分组统计和原生投影
    publication.py        # 配准阶段的原子发布、完整内容身份与恢复校验
    las.py                # LAS 1.2 分块采样与格式检查
    report.py             # 离线点云查看器及原生投影对照
    inspection.py         # 冻结变换、支持重现、独立结构诊断
    diagnostics.py        # 支持拒绝原因、三角化角、重复扫描与观察侧代理
    structure_report.py   # 分区点云定位、真实二维观测与局部原图
  pipeline/
    runner.py             # 阶段顺序、最终 BA、正式导出
    runtime.py            # 工作目录锁、阶段状态与耗时
    task.py               # 冻结执行版本与交付选项
    preflight.py          # 待执行阶段的 GPU、后端、权重与磁盘预检
    data.py               # 全场掩码、特征、候选配对和匹配
    observations.py       # 生产全量观测身份与模式分发
    holdout.py            # 验证模式冻结留出观测
    quality.py            # 生产质量诊断与报告
    registration.py       # 输入冻结、LAS 验收、联合/全局回退选择及正式导出
    calibration.py        # 复用全局特征做局部标定
    reconstruction.py     # 全局建图、轨迹筛选、BA 与报告
  sfm/
    geometry.py           # 两视图几何、紧凑成轨与旋转闭环检查
    tracks.py             # 时序配对与留出抽样
    reconstruction.py     # 数据库、种子模型与有界 BA
    depth_guard.py        # 正深度约束
    track_filter.py       # 训练轨迹一致性筛选
    evaluation.py         # 冻结查询的几何预测评估
    statistics.py         # 成对轨迹统计
```

基础模块不依赖阶段编排。`pipeline/` 组合这些部件；`colmap.py` 可独立使用，读取已有模型和场景即可导出。正式入口不依赖 `work/` 下的实验脚本。原生扩展位于 `native/`，是主工程必需依赖；math/Python 路径仅供显式诊断和旧配置续跑。

## 工作目录

```text
WORK/
  pipeline_config.json     # 场景选择配置及身份
  pipeline_task.json       # 程序/依赖/后端/设备与交付身份
  pipeline_execution_audit.json # 可兼容预检/身份管理代码变更的审计（有变更时生成）
  pipeline_status.json     # 最近一次调用的阶段状态
  run_result.json          # 最近成功调用的结果
  .pipeline.lock           # flock 锁；文件留存不表示进程仍运行
  manifest.json            # 初始输入清单
  images.txt
  fov_masks/
  features/                # 内容寻址的特征缓存
  matches/                 # 按匹配后端区分身份的匹配缓存
  calibration/
    manifest.json
    frontend.json
    split.json             # 仅验证模式
    run_contract.json
    mapping/
    stages/
    calibration_identity.json
    summary.json
  sfm/
    manifest.json
    masks_v2/              # 版本 2 的语义、设备掩码和原因图
    mask_review.html
    frontend.json
    observation_protocol.json  # 生产模式
    holdout_protocol.json      # 以下三项仅验证模式
    split.json
    triangle_validation_split.json
    geometry_protocol.json
    mapping/
    stages/
    optimization_summary.json  # 生产模式
    comparison.json            # 验证模式
    final/sparse/0/        # 视觉原生鱼眼模型；joint 模式的初始化
    final/quality_summary.json     # 生产模式
    final/validation_summary.json  # 验证模式
    index.html
  registration/            # joint 模式
    inputs.json            # LAS/导航内容身份、显式坐标假设与验收设置
    protocol.json          # 视觉源模型、联合配方和冻结门槛
    visual_colmap/         # 内部视觉源交付，含组合掩码
    global/                # 验收的全局配准及内容提交标记
    joint/                 # 纯 BA 对照和 pose + LAS 分支，或约束不足记录
    final/sparse/0/        # 最终选中的 LAS 坐标模型
    final/result.json      # 实际选择、回退原因和上游身份
    final/quality_summary.json
    final/inspection.html
  colmap/                  # 可交付数据集
```

上表展示 local 标定模式；input 模式的标定阶段仅冻结输入内参身份，不生成局部建图与 BA 产物。

工作目录使用绝对路径引用原始数据和缓存，不承诺可搬迁。`colmap/` 的复制模式用于可搬迁交付。

## 运行与恢复

1. `run` 获得 Linux `flock` 排他锁，再写入配置和阶段产物。另一个进程使用相同目录时立即失败。
2. 新目录保存版本 5 配置，默认 production + local + joint + flash；版本 1/2/3 继续按 validation 运行，版本 4 保留其模式，所有旧版本保留 visual 配准模式。续跑加载已保存配置。版本 1 未写匹配后端时解释为 math，显式 flash 不变；旧文件保持原字节。旧配置兼容不绕过数值源码身份检查。命令行传入不同配置、不同输入根目录，或未由新流程管理的非空目录，均拒绝混用。
3. JSON 先写同级临时文件再替换。任务记录全部源码、原生模块、依赖、后端和交付选项，启动及导出前复核。数值相关源码、依赖或后端变化均拒绝继续，并列出变化项；仅不参与拟合的 `pipeline/preflight.py` 和 `pipeline/task.py` 可兼容，原任务身份不改写，变化哈希单独记入 `pipeline_execution_audit.json`。旧任务补建身份时明确记录历史执行版本缺失；旧配置和几何协议保持原字节。特征和匹配的内容哈希继续贯穿标定、全局几何和最终导出。
4. 状态文件只是执行记录。是否能复用结果由实际产物、完成身份和依赖检查决定。
5. 全量重跑会检查并复用完整缓存或阶段；`--stage` 只运行指定阶段。状态标记为 `completed` 仅表示本次请求的阶段完成；`run_result.json` 的 `delivery_complete` 明确表示本次是否发布交付。每次调用有独立 `invocation_id`。
6. Python 异常或 Ctrl-C 会记录失败或中断。进程被强制杀死时可能仍显示 `running`；进程退出后系统会释放锁，重跑将核对实际产物。

局部/全局数据库缺少完整 seed 时，数据库及 SQLite 日志、seed 自动归档后重建。建图与各 BA 子阶段先在同级临时目录写出并回读模型，再发布模型与报告；未提交的旧部分产物归档为 `*.interrupted-*`。失败只重做未完成子阶段，上游完整模型继续复用。
已完成身份对应的文件被修改或丢失时仍拒绝继续；数值失败、质量拒绝不自动降低门槛重试。强制杀死进程可能留下隐藏临时目录，续跑不使用它。导出采用整目录发布。

缓存命中时不加载 LoMa、不解码原图；完整掩码预览复用。新交付记录请求和内容指纹；身份相同的重导出重新验包后直接复用，不重新复制图像或生成掩码。旧交付缺少请求指纹时保持完整比对路径，原身份文件不改写。

总耗时与各阶段均使用 `perf_counter`；`elapsed_seconds = stage_elapsed_seconds + overhead_seconds`。
UTC 只用于显示日期，不用于计时。总耗时范围为阶段执行与预检，不包含 CLI 导入、任务绑定和最终状态文件写入；不能直接当作包含进程启动的 GNU time 计时。成功结果与状态文件保存同一份计时值。

阶段结束后，状态文件的 `resources` 保存 RSS、CPU 时间和 PyTorch CUDA 分配器指标；
`operations` 保存子操作的包含子调用耗时，父子耗时不能相加。
`host_memory_reclaim` 记录空闲堆页回收前后的 RSS；指标口径及缓存预算见 [资源报告](resources.md)。

## 固定的数值流程

保留已实跑的视觉流程：1536 检测尺寸、每图最多 3072 个 LoMa 特征，时序加描述子检索候选配对；生产模式使用全部质量筛选后的观测；验证模式在所有几何拟合前冻结全局留出；局部依次优化固定内参、焦距与低阶径向、全部径向、主点；全局依次固定标定 BA、训练轨迹筛选 BA、共享标定 BA，最后执行最多 40 次迭代的有界 BA。

相机共享内参，每次曝光使用独立 frame。正深度与 160° 单调投影检查、训练覆盖检查、全部图像注册要求均保留。标定片段选择属于配置；不根据留出误差搜索参数或选择模型。

生产与验证模式的差异和来源契约见 [模式说明](production_pipeline.md)。

## 当前边界

- 输入适配 Geosun 的 `.pro`、CameraPos 和 L/R 原图命名；设备遮挡规则针对当前设备。
- SfM 无人工标注依赖，掩码预览只供检查，不阻塞流程。
- 导出保持原生 `OPENCV_FISHEYE`；使用方需要支持该相机模型。
- 视觉中间结果为任意尺度。默认 joint 模式另行完成 LAS 配准与固定内参联合 BA，正式交付位于输入 LAS 的 XYZ 框架。当前米制单位与 ENH 原点是显式假设，未经独立地理精度验收；原始 LAS 不改变。
- 数值与阶段测试包含真实 PyCOLMAP 模型读写和小模型 BA；管线集成测试对大规模建图与 BA 使用替身。
- 数值配方最近一次完整运行见 [生产报告](production_pipeline.md)：20 分 31.30 秒，未达到 20 分钟目标；本轮流程改进未重测完整冷启动，也未验证下游渲染质量。历史性能、资源与消融记录见 [验证摘要索引](benchmarks/README.md)。

## 当前流程验证（2026-09-30）

- 89 项测试通过，覆盖缓存不加载模型、中断发布与续跑、已提交模型损坏拒绝、交付选项继承、重复导出复用、质量门槛和统一计时；预检修改可以审计后续跑，数值源码、依赖及后端变化仍拒绝。
- 既有真实篮球场交付包完整内容检查通过：536 张图、75,893 个点，原图复制并包含组合掩码。
- 在隔离副本中使用当前程序导出真实模型（原图链接、不含掩码），省略交付参数再次导出，随后独立 `check` 通过。任务、交付身份与模型文件的内容及修改时间均未变化；原工作目录 41 个配置、协议和身份文件保持原字节。
- 真实质量报告通过新篮球场整体及 L/R 残差、内参触界门槛；成功结果与状态文件的调用身份和计时一致。本轮验证范围为已有模型的导出、续跑及验包，未执行全量 GPU 前端或冷启动重建。
- `work/basketball_industrial_01` 的重建已完成，但运行期间补充预检权重列表曾使旧任务检查在导出前拒绝。修复后只补执行 export，536 张图、75,749 点的原图、掩码及 PLY 交付独立验包通过；81 个原模型与冻结产物的哈希保持一致，原失败状态和恢复验证保存在该目录的 `export_recovery/` 中。

可追溯摘要见 [industrial_pipeline_20260930.json](benchmarks/industrial_pipeline_20260930.json)，
启动与隔离续跑的原始验证记录另保存在 `work/industrial_validation_20260930/`。

## 历史重构验证（2026-09-28）

- 39 项测试通过，包含小模型实际 BA、留出隔离、配置冻结、工作目录锁、失败记录与 COLMAP 迁移校验。
- 依赖锁检查通过，0.2.0 wheel 和源码包构建成功；源码包包含配置、文档与测试，不包含运行数据。
- wheel 安装到独立目录，从源码目录外执行命令。真实篮球场 `prepare` 选出 536 张图和 50 张标定图。
- 新 `check` 对此前完整运行的交付包执行内容校验通过：536 张图、61,495 个点，包含组合掩码。
- 2026-09-28 该轮未重新执行完整 GPU 掩码推理、特征匹配和全场重建；不据此声称重构后的数值结果已经全量复现。
