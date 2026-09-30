#pragma once

#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include "ceres/cost_function.h"

namespace geosunpre {

// Eigen quaternion order (x,y,z,w), matching COLMAP and pyceres.
inline void Rotation(const double* q, double R[3][3], double D[4][3][3]) {
  const double x=q[0], y=q[1], z=q[2], w=q[3];
  const double matrix[3][3] = {
    {1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)},
    {2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)},
    {2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)}};
  const double derivative[4][3][3] = {
    {{0,2*y,2*z},{2*y,-4*x,-2*w},{2*z,2*w,-4*x}},
    {{-4*y,2*x,2*w},{2*x,0,2*z},{-2*w,2*z,-4*y}},
    {{-4*z,-2*w,2*x},{2*w,-4*z,2*y},{2*x,2*y,0}},
    {{0,-2*z,2*y},{2*z,0,-2*x},{-2*y,2*x,0}}};
  for (int a=0;a<3;++a) for (int b=0;b<3;++b) {
    R[a][b]=matrix[a][b];
    if (D) for (int k=0;k<4;++k) D[k][a][b]=derivative[k][a][b];
  }
}

inline void CheckVector(const std::array<double,3>& v) {
  for (double x : v) if (!std::isfinite(x)) throw std::invalid_argument("Expected a finite vector");
}

// S(X) = exp(log_scale) R(q) X + t. Separate q[4] and t_log_scale[4]
// parameter blocks allow the existing EigenQuaternionManifold to be reused.
class Sim3PointPlaneCost final : public ceres::CostFunction {
 public:
  Sim3PointPlaneCost(std::array<double,3> centre, std::array<double,3> normal,
                    double gain) : centre_(centre), normal_(normal), gain_(gain) {
    CheckVector(centre); CheckVector(normal);
    const double length=std::sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
    if (!std::isfinite(gain) || gain<=0 || std::abs(length-1)>1e-6)
      throw std::invalid_argument("Expected a positive gain and unit normal");
    set_num_residuals(1); *mutable_parameter_block_sizes()={3,4,4};
  }
  bool Evaluate(double const* const* p, double* r, double** j) const override {
    double R[3][3], D[4][3][3]; Rotation(p[1],R,D);
    const double scale=std::exp(p[2][3]);
    double rotated[3]={0,0,0};
    for (int a=0;a<3;++a) for (int b=0;b<3;++b) rotated[a]+=R[a][b]*p[0][b];
    r[0]=0;
    for (int a=0;a<3;++a) r[0]+=gain_*normal_[a]*(scale*rotated[a]+p[2][a]-centre_[a]);
    if (j) {
      if (j[0]) {
        std::fill(j[0],j[0]+3,0.);
        for (int b=0;b<3;++b) for (int a=0;a<3;++a) j[0][b]+=gain_*scale*normal_[a]*R[a][b];
      }
      if (j[1]) {
        std::fill(j[1],j[1]+4,0.);
        for (int k=0;k<4;++k) for (int a=0;a<3;++a) for (int b=0;b<3;++b)
          j[1][k]+=gain_*scale*normal_[a]*D[k][a][b]*p[0][b];
      }
      if (j[2]) {
        for (int a=0;a<3;++a) j[2][a]=gain_*normal_[a];
        j[2][3]=0;
        for (int a=0;a<3;++a) j[2][3]+=gain_*scale*normal_[a]*rotated[a];
      }
    }
    return std::isfinite(r[0]);
  }
 private:
  const std::array<double,3> centre_, normal_;
  const double gain_;
};

class Sim3PositionPriorCost final : public ceres::CostFunction {
 public:
  Sim3PositionPriorCost(std::array<double,3> position, double gain)
      : position_(position), gain_(gain) {
    CheckVector(position);
    if (!std::isfinite(gain) || gain<=0) throw std::invalid_argument("Expected a positive gain");
    set_num_residuals(3); *mutable_parameter_block_sizes()={7,4,4};
  }
  bool Evaluate(double const* const* p, double* r, double** j) const override {
    double camera_R[3][3], camera_D[4][3][3], R[3][3], D[4][3][3];
    Rotation(p[0],camera_R,camera_D); Rotation(p[1],R,D);
    const double scale=std::exp(p[2][3]);
    double centre[3]={0,0,0}, rotated[3]={0,0,0};
    for (int b=0;b<3;++b) for (int a=0;a<3;++a) centre[b]-=camera_R[a][b]*p[0][4+a];
    for (int a=0;a<3;++a) for (int b=0;b<3;++b) rotated[a]+=R[a][b]*centre[b];
    for (int a=0;a<3;++a) r[a]=gain_*(scale*rotated[a]+p[2][a]-position_[a]);
    if (j) {
      if (j[0]) {
        std::fill(j[0],j[0]+21,0.);
        for (int a=0;a<3;++a) {
          for (int k=0;k<4;++k) for (int b=0;b<3;++b) for (int c=0;c<3;++c)
            j[0][a*7+k]-=gain_*scale*R[a][b]*camera_D[k][c][b]*p[0][4+c];
          for (int k=0;k<3;++k) for (int b=0;b<3;++b)
            j[0][a*7+4+k]-=gain_*scale*R[a][b]*camera_R[k][b];
        }
      }
      if (j[1]) {
        std::fill(j[1],j[1]+12,0.);
        for (int a=0;a<3;++a) for (int k=0;k<4;++k) for (int b=0;b<3;++b)
          j[1][a*4+k]+=gain_*scale*D[k][a][b]*centre[b];
      }
      if (j[2]) {
        std::fill(j[2],j[2]+12,0.);
        for (int a=0;a<3;++a) {
          j[2][a*4+a]=gain_;
          j[2][a*4+3]=gain_*scale*rotated[a];
        }
      }
    }
    return std::all_of(r,r+3,[](double value){return std::isfinite(value);});
  }
 private:
  const std::array<double,3> position_;
  const double gain_;
};

}  // namespace geosunpre
