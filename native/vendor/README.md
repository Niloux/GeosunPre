Vendored from Ceres Solver tag 2.2.0:

- include/ceres/cost_function.h
- internal/ceres/cost_function.cc

Source: https://github.com/ceres-solver/ceres-solver/tree/2.2.0

BSD license retained in each file. The header only removes DLL export and compiler warning includes/macros. Class layout and constructor/destructor are unchanged. This extension supplies only the CostFunction interface, not another solver. It uses the existing pyceres 2.6 type registration and its Ceres 2.2.0 ABI; pybind11 is pinned to the matching 3.0.1. Other Ceres versions are rejected at import.
