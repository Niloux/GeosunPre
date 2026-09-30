#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "ceres/cost_function.h"
#include "joint_costs.h"

namespace py = pybind11;

class PositiveDepthCost final : public ceres::CostFunction {
 public:
  explicit PositiveDepthCost(double depth) : minimum(.05 * depth), scale(1000 / depth) {
    if (!std::isfinite(depth) || depth <= 1e-9)
      throw std::invalid_argument("Initial depth must be finite and positive");
    set_num_residuals(1);
    *mutable_parameter_block_sizes() = {3, 7};
  }
  bool Evaluate(double const* const* p, double* r, double** j) const override {
    const double X=p[0][0], Y=p[0][1], Z=p[0][2];
    const double x=p[1][0], y=p[1][1], z=p[1][2], w=p[1][3];
    const double row[3]={2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)};
    const double depth=row[0]*X+row[1]*Y+row[2]*Z+p[1][6];
    const bool active=depth<minimum;
    r[0]=active ? (minimum-depth)*scale : 0.;
    if (j) {
      if (j[0]) for (int k=0;k<3;++k) j[0][k]=active ? -scale*row[k] : 0.;
      if (j[1]) {
        std::fill(j[1], j[1]+7, 0.);
        if (active) {
          j[1][0]=-scale*(2*z*X+2*w*Y-4*x*Z);
          j[1][1]=-scale*(-2*w*X+2*z*Y-4*y*Z);
          j[1][2]=-scale*(2*x*X+2*y*Y);
          j[1][3]=-scale*(-2*y*X+2*x*Y);
          j[1][6]=-scale;
        }
      }
    }
    return true;
  }
  const double minimum, scale;
};

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
