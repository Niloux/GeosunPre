#include "geometry/camera.h"
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>

namespace geosun {
Camera Camera::FromJson(const Json& v) {
  Camera c;
  c.label = v.at("label");
  c.width = v.at("width");
  c.height = v.at("height");
  c.params = v.at("params").get<std::array<double, 8>>();
  c.principal_offset = v.at("principal_offset_px").get<std::array<double, 2>>();
  return c;
}
Json Camera::ToJson() const {
  return {{"label", label},
          {"width", width},
          {"height", height},
          {"params", params},
          {"principal_offset_px", principal_offset}};
}
double Camera::Radius(double fov) const {
  if (width <= 0 || height <= 0 || params[0] <= 0 || params[1] <= 0 ||
      !std::all_of(params.begin(), params.end(), [](double x) { return std::isfinite(x); }))
    throw std::runtime_error(label + ": invalid camera calibration");
  const double limit = fov * 3.14159265358979323846 / 360;
  for (int j = 0; j < 256; ++j) {
    const double t = limit * j / 255, t2 = t * t;
    double derivative = 1, power = t2;
    for (int k = 0; k < 4; ++k) {
      derivative += (2 * k + 3) * params[4 + k] * power;
      power *= t2;
    }
    if (derivative <= 0)
      throw std::runtime_error(label + ": nonmonotonic fisheye calibration");
  }
  const double t2 = limit * limit;
  return limit * (1 + t2 * (params[4] + t2 * (params[5] + t2 * (params[6] + t2 * params[7]))));
}
bool Camera::Valid(double x, double y, double fov) const {
  const double dx = (x - params[2]) / params[0], dy = (y - params[3]) / params[1], r = Radius(fov);
  return std::isfinite(x) && std::isfinite(y) && x >= 0 && y >= 0 && x < width && y < height &&
         dx * dx + dy * dy <= r * r;
}
cv::Mat Camera::Mask(double fov) const {
  cv::Mat out(height, width, CV_8UC1);
  const double r = Radius(fov);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x) {
      const double dx = (x + .5 - params[2]) / params[0], dy = (y + .5 - params[3]) / params[1];
      out.at<uint8_t>(y, x) = dx * dx + dy * dy <= r * r ? 255 : 0;
    }
  return out;
}
Eigen::Vector2d Camera::Project(const Eigen::Vector3d& v) const {
  const double r = std::hypot(v.x(), v.y());
  if (r < 1e-15)
    return {params[2], params[3]};
  const double t = std::atan2(r, v.z()), t2 = t * t;
  const double d =
      t * (1 + t2 * (params[4] + t2 * (params[5] + t2 * (params[6] + t2 * params[7])))) / r;
  return {params[0] * v.x() * d + params[2], params[1] * v.y() * d + params[3]};
}
Eigen::Matrix<double, 2, 3> Camera::ProjectionJacobian(const Eigen::Vector3d& v) const {
  Eigen::Matrix<double, 2, 3> j = Eigen::Matrix<double, 2, 3>::Zero();
  const double r = std::hypot(v.x(), v.y());
  if (r < 1e-10) {
    j(0, 0) = params[0] / v.z();
    j(1, 1) = params[1] / v.z();
    return j;
  }
  const double theta = std::atan2(r, v.z()), t2 = theta * theta;
  const double radial =
      theta * (1 + t2 * (params[4] + t2 * (params[5] + t2 * (params[6] + t2 * params[7]))));
  const double slope =
      1 + t2 * (3 * params[4] + t2 * (5 * params[5] + t2 * (7 * params[6] + t2 * 9 * params[7])));
  const Eigen::Vector3d dr(v.x() / r, v.y() / r, 0);
  const Eigen::Vector3d dt = (v.z() * dr - Eigen::Vector3d(0, 0, r)) / v.squaredNorm();
  const double scale = radial / r;
  const Eigen::Vector3d ds = (slope * dt - scale * dr) / r;
  j.row(0) = params[0] * (v.x() * ds + Eigen::Vector3d(scale, 0, 0)).transpose();
  j.row(1) = params[1] * (v.y() * ds + Eigen::Vector3d(0, scale, 0)).transpose();
  return j;
}
Eigen::Vector3d Camera::Ray(const Eigen::Vector2d& pixel) const {
  const double x = (pixel.x() - params[2]) / params[0], y = (pixel.y() - params[3]) / params[1],
               r = std::hypot(x, y);
  if (r < 1e-15)
    return Eigen::Vector3d::UnitZ();
  double t = r;
  for (int i = 0; i < 30; ++i) {
    const double t2 = t * t;
    const double f =
        t * (1 + t2 * (params[4] + t2 * (params[5] + t2 * (params[6] + t2 * params[7])))) - r;
    const double df =
        1 + t2 * (3 * params[4] + t2 * (5 * params[5] + t2 * (7 * params[6] + t2 * 9 * params[7])));
    if (df <= 0)
      throw std::runtime_error("Fisheye inversion outside monotonic domain");
    const double delta = f / df;
    t -= delta;
    if (std::abs(delta) < 1e-12)
      break;
  }
  return {x * std::sin(t) / r, y * std::sin(t) / r, std::cos(t)};
}
std::array<double, 8> Camera::CalibrationBounds() const {
  return {params[0] * .1, params[1] * .1, 50, 50, .05, .02, .01, .005};
}
Eigen::Quaterniond Pose::Rotation() const {
  return Eigen::Quaterniond(params[3], params[0], params[1], params[2]);
}
Eigen::Vector3d Pose::Translation() const { return {params[4], params[5], params[6]}; }
Eigen::Vector3d Pose::Center() const { return -(Rotation().conjugate() * Translation()); }
Eigen::Vector3d Pose::Apply(const Eigen::Vector3d& p) const {
  return Rotation() * p + Translation();
}
void Pose::Set(const Eigen::Quaterniond& q, const Eigen::Vector3d& t) {
  const auto unit = q.normalized();
  params = {unit.x(), unit.y(), unit.z(), unit.w(), t.x(), t.y(), t.z()};
}
Eigen::Vector3d Similarity::Apply(const Eigen::Vector3d& p) const {
  return scale * (rotation * p) + translation;
}
Json Similarity::ToJson() const {
  return {{"scale", scale},
          {"quaternion_xyzw", {rotation.x(), rotation.y(), rotation.z(), rotation.w()}},
          {"translation", {translation.x(), translation.y(), translation.z()}}};
}
Similarity Similarity::FromJson(const Json& v) {
  const auto q = v.at("quaternion_xyzw").get<std::array<double, 4>>();
  const auto t = v.at("translation").get<std::array<double, 3>>();
  return {v.at("scale"), Eigen::Quaterniond(q[3], q[0], q[1], q[2]),
          Eigen::Vector3d(t[0], t[1], t[2])};
}
Similarity FitSimilarity(const std::vector<Eigen::Vector3d>& a,
                         const std::vector<Eigen::Vector3d>& b) {
  if (a.size() != b.size() || a.size() < 3)
    throw std::runtime_error("Similarity requires at least three correspondences");
  Eigen::Matrix<double, 3, Eigen::Dynamic> x(3, a.size()), y(3, b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    x.col(i) = a[i];
    y.col(i) = b[i];
  }
  const Eigen::Matrix4d m = Eigen::umeyama(x, y, true);
  const double s = std::cbrt(m.topLeftCorner<3, 3>().determinant());
  if (!m.allFinite() || s <= 0)
    throw std::runtime_error("Degenerate similarity fit");
  return {s, Eigen::Quaterniond(m.topLeftCorner<3, 3>() / s).normalized(),
          m.topRightCorner<3, 1>()};
}
double Quantile(std::vector<double> v, double f) {
  if (v.empty())
    return INFINITY;
  const double at = f * (v.size() - 1);
  const size_t i = size_t(at), j = std::min(i + 1, v.size() - 1);
  std::nth_element(v.begin(), v.begin() + i, v.end());
  const double low = v[i];
  if (i == j)
    return low;
  std::nth_element(v.begin() + i + 1, v.begin() + j, v.end());
  return low + (at - i) * (v[j] - low);
}
Json Statistics(const std::vector<double>& v) {
  if (v.empty())
    return {{"count", 0}, {"median", nullptr}, {"p90", nullptr}};
  return {{"count", v.size()}, {"median", Quantile(v, .5)}, {"p90", Quantile(v, .9)}};
}
}  // namespace geosun
