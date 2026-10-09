# 原生 BA 约束

`geosunpre-native` 是主工程必需的本地扩展，由 `uv sync --locked` 构建安装。需要 C++17 编译器及 Python 开发头文件，pybind11 固定为 3.0.1。

扩展实现 `PositiveDepthCost` 的残差和解析 Jacobian，支持四线程受限视觉 BA。`Sim3PointPlaneCost` 与 `Sim3PositionPriorCost` 为联合优化提供 LAS 点到面及相机中心位置先验，Jacobian 在 C++ 中计算。固定兼容 pyceres 2.6 / Ceres 2.2.0，导入检查 Ceres 版本，当前实现版本为 2。修改相关版本须重新核对 ABI 并执行数值及实际求解检查。

全局变换采用 Eigen 顺序四元数 `[x,y,z,w]` 与 `[tx,ty,tz,log_scale]` 两个参数块。相机中心由 `cam_from_world` 的 `-R.T @ t` 计算，输入姿态四元数不参与求解。`joint_costs.h` 是增量编译依赖并随源码包发布。

```bash
# 主工程重建本地扩展
uv sync --locked --reinstall-package geosunpre-native

# 独立发行包
uv build native

```

没有 Python 深度约束回退。缺失联合约束时会要求重建扩展。`vendor/` 仅包含 Ceres CostFunction 接口及构造/析构实现，不包含另一份求解器；来源见 [vendor/README.md](vendor/README.md)，许可证见 [vendor/LICENSE.ceres](vendor/LICENSE.ceres)，随源码包和 wheel 分发。
