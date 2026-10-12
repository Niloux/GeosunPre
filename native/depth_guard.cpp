#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "ceres/cost_function.h"
#include "joint_costs.h"
#include "depth_cost.h"

namespace py = pybind11;

using geosunpre::PositiveDepthCost;

PYBIND11_MODULE(geosunpre_native, m) {
  auto ceres = py::module_::import("pyceres");
  if (py::str(ceres.attr("__ceres_version__")).cast<std::string>() != "2.2.0")
    throw std::runtime_error("geosunpre_native requires Ceres 2.2.0 (pyceres 2.6)");
  m.attr("implementation_version") = 2;
  py::classh<PositiveDepthCost, ceres::CostFunction>(m, "PositiveDepthCost")
    .def(py::init<double>())
    .def_readonly("minimum", &PositiveDepthCost::minimum)
    .def_readonly("scale", &PositiveDepthCost::scale)
    .def("evaluate_arrays", [](const PositiveDepthCost& cost,
          py::array_t<double, py::array::c_style | py::array::forcecast> point,
          py::array_t<double, py::array::c_style | py::array::forcecast> pose) {
      if (point.size()!=3 || pose.size()!=7) throw std::invalid_argument("Expected point[3], pose[7]");
      py::array_t<double> r(1), jp(3), jq(7);
      const double* p[]={point.data(), pose.data()};
      double* j[]={jp.mutable_data(), jq.mutable_data()};
      cost.Evaluate(p, r.mutable_data(), j);
      return py::make_tuple(r,jp,jq);
    });
  auto evaluate = [](const ceres::CostFunction& cost,
        py::array_t<double, py::array::c_style | py::array::forcecast> variable,
        py::array_t<double, py::array::c_style | py::array::forcecast> quaternion,
        py::array_t<double, py::array::c_style | py::array::forcecast> translation_scale) {
    const auto& sizes=cost.parameter_block_sizes();
    if (variable.size()!=sizes[0] || quaternion.size()!=4 || translation_scale.size()!=4)
      throw std::invalid_argument("Expected variable[3 or 7], quaternion[4], translation_log_scale[4]");
    const int rows=cost.num_residuals();
    py::array_t<double> r(rows), jv({rows,sizes[0]}), jq({rows,4}), jt({rows,4});
    const double* p[]={variable.data(),quaternion.data(),translation_scale.data()};
    double* j[]={jv.mutable_data(),jq.mutable_data(),jt.mutable_data()};
    if (!cost.Evaluate(p,r.mutable_data(),j)) throw std::invalid_argument("Nonfinite residual");
    return py::make_tuple(r,jv,jq,jt);
  };
  py::classh<geosunpre::Sim3PointPlaneCost, ceres::CostFunction>(m,"Sim3PointPlaneCost")
    .def(py::init<std::array<double,3>,std::array<double,3>,double>())
    .def("evaluate_arrays",evaluate);
  py::classh<geosunpre::Sim3PositionPriorCost, ceres::CostFunction>(m,"Sim3PositionPriorCost")
    .def(py::init<std::array<double,3>,double>())
    .def("evaluate_arrays",evaluate);
}
