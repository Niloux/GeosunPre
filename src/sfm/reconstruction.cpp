#include "sfm/reconstruction.h"
#include "sfm/optimization.h"
#include "io/cache.h"
#include <opencv2/calib3d.hpp>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include <atomic>
#include <thread>
#include <numeric>
#include <unordered_map>
#include <set>
#include <random>

namespace geosun {
namespace {
Eigen::Matrix3d Matrix(const cv::Mat& input) {
  cv::Mat m;
  input.convertTo(m, CV_64F);
  Eigen::Matrix3d out;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      out(r, c) = m.at<double>(r, c);
  return out;
}
// Polynomials in x, y, z of degree at most 3; monomials ordered 1, x, y, z, then degree 2 and 3.
struct Cubic {
  std::array<double, 20> c{};
};
struct Monomials {
  std::array<std::array<int, 3>, 20> exponent{};
  int index[4][4][4];
  int count[4];  // monomials of degree below d, for d = 1..3 ending at 4, 10, 20
  Monomials() {
    int n = 0;
    for (int degree = 0; degree <= 3; ++degree) {
      for (int a = degree; a >= 0; --a)
        for (int b = degree - a; b >= 0; --b) {
          exponent[n] = {a, b, degree - a - b};
          index[a][b][degree - a - b] = n++;
        }
      count[degree] = n;
    }
  }
};
const Monomials monomials;
// Product of a polynomial of degree da with one of degree db, da + db <= 3.
Cubic Multiply(const Cubic& p, int da, const Cubic& q, int db) {
  Cubic r;
  for (int i = 0; i < monomials.count[da]; ++i) {
    if (p.c[i] == 0)
      continue;
    const auto& u = monomials.exponent[i];
    for (int j = 0; j < monomials.count[db]; ++j) {
      const auto& v = monomials.exponent[j];
      r.c[monomials.index[u[0] + v[0]][u[1] + v[1]][u[2] + v[2]]] += p.c[i] * q.c[j];
    }
  }
  return r;
}
Cubic Combine(const Cubic& p, double a, const Cubic& q, double b) {
  Cubic r;
  for (int i = 0; i < 20; ++i)
    r.c[i] = a * p.c[i] + b * q.c[i];
  return r;
}
// Essential matrices with b^T E a = 0 for five perspective correspondences, unit Frobenius norm:
// E = xX + yY + zZ + W spans the null space of the epipolar constraints; det(E) = 0 and
// 2 E E^T E - tr(E E^T) E = 0 give ten cubics, whose degree-3 monomials are eliminated to an
// action matrix for multiplication by x. Its real eigenvectors are the solutions (Stewenius).
std::vector<Eigen::Matrix3d> FivePoint(const std::vector<cv::Point2d>& a,
                                       const std::vector<cv::Point2d>& b) {
  Eigen::Matrix<double, 5, 9> q;
  for (int i = 0; i < 5; ++i)
    q.row(i) << b[i].x * a[i].x, b[i].x * a[i].y, b[i].x, b[i].y * a[i].x, b[i].y * a[i].y,
        b[i].y, a[i].x, a[i].y, 1;
  const Eigen::JacobiSVD<Eigen::Matrix<double, 5, 9>> svd(q, Eigen::ComputeFullV);
  const Eigen::Matrix<double, 9, 4> basis = svd.matrixV().rightCols<4>();
  std::array<Cubic, 9> e;
  for (int k = 0; k < 9; ++k) {
    e[k].c[1] = basis(k, 0);
    e[k].c[2] = basis(k, 1);
    e[k].c[3] = basis(k, 2);
    e[k].c[0] = basis(k, 3);
  }
  auto at = [&](int r, int c) -> const Cubic& { return e[3 * r + c]; };
  std::array<Cubic, 10> equations;
  // det(E) by the first row.
  const auto minor = [&](int r0, int c0, int r1, int c1) {
    return Combine(Multiply(at(r0, c0), 1, at(r1, c1), 1), 1,
                   Multiply(at(r0, c1), 1, at(r1, c0), 1), -1);
  };
  equations[0] = Combine(Combine(Multiply(at(0, 0), 1, minor(1, 1, 2, 2), 2), 1,
                                 Multiply(at(0, 1), 1, minor(1, 0, 2, 2), 2), -1),
                         1, Multiply(at(0, 2), 1, minor(1, 0, 2, 1), 2), 1);
  std::array<Cubic, 9> eet;
  for (int i = 0; i < 3; ++i)
    for (int k = 0; k < 3; ++k)
      for (int j = 0; j < 3; ++j)
        eet[3 * i + k] = Combine(eet[3 * i + k], 1, Multiply(at(i, j), 1, at(k, j), 1), 1);
  const Cubic trace = Combine(Combine(eet[0], 1, eet[4], 1), 1, eet[8], 1);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      Cubic sum;
      for (int k = 0; k < 3; ++k)
        sum = Combine(sum, 1, Multiply(eet[3 * i + k], 2, at(k, j), 1), 1);
      equations[1 + 3 * i + j] = Combine(sum, 2, Multiply(trace, 2, at(i, j), 1), -1);
    }
  // Columns: the ten cubic monomials, then the basis x^2, xy, xz, y^2, yz, z^2, x, y, z, 1.
  Eigen::Matrix<double, 10, 10> cubic, rest;
  for (int r = 0; r < 10; ++r)
    for (int k = 0; k < 10; ++k) {
      cubic(r, k) = equations[r].c[10 + k];
      rest(r, k) = k < 6 ? equations[r].c[4 + k] : equations[r].c[k == 9 ? 0 : k - 5];
    }
  const Eigen::FullPivLU<Eigen::Matrix<double, 10, 10>> lu(cubic);
  if (!lu.isInvertible())
    return {};
  const Eigen::Matrix<double, 10, 10> reduced = lu.solve(rest);
  // x times each basis monomial: x^3, x^2y, x^2z, xy^2, xyz, xz^2 are cubic, the rest are basis.
  const std::array<std::array<int, 3>, 6> raised{
      {{3, 0, 0}, {2, 1, 0}, {2, 0, 1}, {1, 2, 0}, {1, 1, 1}, {1, 0, 2}}};
  Eigen::Matrix<double, 10, 10> action = Eigen::Matrix<double, 10, 10>::Zero();
  for (int i = 0; i < 6; ++i)
    action.row(i) = -reduced.row(
        monomials.index[raised[i][0]][raised[i][1]][raised[i][2]] - 10);
  action(6, 0) = action(7, 1) = action(8, 2) = action(9, 6) = 1;
  const Eigen::EigenSolver<Eigen::Matrix<double, 10, 10>> eigen(action);
  if (eigen.info() != Eigen::Success)
    return {};
  std::vector<Eigen::Matrix3d> solutions;
  for (int s = 0; s < 10; ++s) {
    if (std::abs(eigen.eigenvalues()[s].imag()) > 1e-10)
      continue;
    const Eigen::Matrix<double, 10, 1> v = eigen.eigenvectors().col(s).real();
    if (std::abs(v[9]) < 1e-10)
      continue;
    const Eigen::Matrix<double, 9, 1> flat =
        basis.col(0) * (v[6] / v[9]) + basis.col(1) * (v[7] / v[9]) + basis.col(2) * (v[8] / v[9]) +
        basis.col(3);
    Eigen::Matrix3d candidate;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        candidate(r, c) = flat[3 * r + c];
    solutions.push_back(candidate / candidate.norm());
  }
  return solutions;
}
bool Triangulate(const Pose& a, const Pose& b, const Eigen::Vector3d& ra, const Eigen::Vector3d& rb,
                 Eigen::Vector3d& point) {
  Eigen::Matrix<double, 3, 4> pa, pb;
  pa.leftCols<3>() = a.Rotation().toRotationMatrix();
  pa.col(3) = a.Translation();
  pb.leftCols<3>() = b.Rotation().toRotationMatrix();
  pb.col(3) = b.Translation();
  Eigen::Matrix4d design;
  design.row(0) = ra.x() * pa.row(2) - ra.z() * pa.row(0);
  design.row(1) = ra.y() * pa.row(2) - ra.z() * pa.row(1);
  design.row(2) = rb.x() * pb.row(2) - rb.z() * pb.row(0);
  design.row(3) = rb.y() * pb.row(2) - rb.z() * pb.row(1);
  const auto v = design.jacobiSvd(Eigen::ComputeFullV).matrixV().col(3).eval();
  if (std::abs(v[3]) < 1e-12)
    return false;
  point = v.head<3>() / v[3];
  return point.allFinite() && a.Apply(point).z() > 0 && b.Apply(point).z() > 0;
}
struct Pair {
  uint32_t a, b;
  Pose relative;
  std::vector<Match> inliers;
  double angle = 0;
};
struct Measurement {
  Eigen::Vector3d ray;
  Eigen::Matrix<double, 3, 2> derivative;
};
Measurement Measure(const Camera& camera, const Eigen::Vector2d& pixel) {
  Measurement m;
  m.ray = camera.Ray(pixel);
  const auto& p = camera.params;
  const Eigen::Vector2d xy((pixel.x() - p[2]) / p[0], (pixel.y() - p[3]) / p[1]);
  const double r = xy.norm();
  if (r < 1e-7) {
    m.derivative.setZero();
    m.derivative(0, 0) = 1 / p[0];
    m.derivative(1, 1) = 1 / p[1];
    return m;
  }
  const double theta = std::atan2(m.ray.head<2>().norm(), m.ray.z()), t2 = theta * theta;
  const double slope = 1 + t2 * (3 * p[4] + t2 * (5 * p[5] + t2 * (7 * p[6] + t2 * 9 * p[7])));
  const double sine = std::sin(theta), cosine = std::cos(theta);
  m.derivative.topRows<2>() = Eigen::Matrix2d::Identity() * (sine / r) +
                              (cosine / (slope * r * r) - sine / (r * r * r)) * xy * xy.transpose();
  m.derivative.row(2) = -sine / (slope * r) * xy.transpose();
  m.derivative.col(0) /= p[0];
  m.derivative.col(1) /= p[1];
  return m;
}
double ExposureTime(const Exposure& e) { return e.week * 604800. + e.seconds; }
double RotationAngleDeg(const Eigen::Matrix3d& R) {
  return std::acos(std::clamp((R.trace() - 1.) * .5, -1., 1.)) * 180. / Pi;
}
Eigen::Matrix3d CrossMatrix(const Eigen::Vector3d& v) {
  Eigen::Matrix3d m;
  m << 0, -v.z(), v.y(), v.z(), 0, -v.x(), -v.y(), v.x(), 0;
  return m;
}
Eigen::Matrix3d ExpSO3(const Eigen::Vector3d& w) {
  const double theta = w.norm();
  if (theta < 1e-12)
    return Eigen::Matrix3d::Identity() + CrossMatrix(w);
  return Eigen::AngleAxisd(theta, w / theta).toRotationMatrix();
}
double EpipolarSigned(const Eigen::Matrix3d& e, const Measurement& a, const Measurement& b) {
  const Eigen::Vector3d ea = e * a.ray, eb = e.transpose() * b.ray;
  const double value = b.ray.dot(ea);
  const double gradient =
      (a.derivative.transpose() * eb).squaredNorm() + (b.derivative.transpose() * ea).squaredNorm();
  return value / std::sqrt(std::max(gradient, 1e-24));
}
double EpipolarSquared(const Eigen::Matrix3d& e, const Measurement& a, const Measurement& b) {
  const double s = EpipolarSigned(e, a, b);
  return s * s;
}
bool RefitEssential(const std::vector<Measurement>& a, const std::vector<Measurement>& b,
                    const Eigen::Matrix3d& current, Eigen::Matrix3d& candidate) {
  Eigen::Matrix<double, 9, 9> normal = Eigen::Matrix<double, 9, 9>::Zero();
  size_t count = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (EpipolarSquared(current, a[i], b[i]) > 9)
      continue;
    const double gradient =
        (a[i].derivative.transpose() * current.transpose() * b[i].ray).squaredNorm() +
        (b[i].derivative.transpose() * current * a[i].ray).squaredNorm();
    Eigen::Matrix<double, 9, 1> row;
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        row[3 * r + c] = b[i].ray[r] * a[i].ray[c];
    normal.noalias() += row * row.transpose() / std::max(gradient, 1e-24);
    ++count;
  }
  if (count < 8)
    return false;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> solve(normal);
  if (solve.info() != Eigen::Success || solve.eigenvalues()[1] <= solve.eigenvalues()[8] * 1e-10)
    return false;
  const Eigen::Matrix<double, 9, 1> v = solve.eigenvectors().col(0);
  Eigen::Matrix3d e;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      e(r, c) = v[3 * r + c];
  const auto svd = e.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
  candidate = svd.matrixU() * Eigen::Vector3d(1, 1, 0).asDiagonal() * svd.matrixV().transpose();
  return candidate.allFinite();
}
Pair Verify(const Scene& scene, const Reconstruction& model, const MatchSet& matches) {
  Pair result{matches.a, matches.b};
  if (matches.matches.size() < 24)
    return result;
  const auto& ca = model.cameras[scene.images[matches.a].camera];
  const auto& cb = model.cameras[scene.images[matches.b].camera];
  std::vector<cv::Point2d> na, nb;
  std::vector<Measurement> ra, rb;
  for (const auto& m : matches.matches) {
    const auto x = Measure(ca, model.images[matches.a].pixels.at(m.a)),
               y = Measure(cb, model.images[matches.b].pixels.at(m.b));
    ra.push_back(x);
    rb.push_back(y);
    na.emplace_back(x.ray.x() / x.ray.z(), x.ray.y() / x.ray.z());
    nb.emplace_back(y.ray.x() / y.ray.z(), y.ray.y() / y.ray.z());
  }
  const double focal = (ca.params[0] + ca.params[1] + cb.params[0] + cb.params[1]) / 4;
  Eigen::Matrix3d essential;
  size_t best = 0;
  double best_cost = INFINITY;
  auto score = [&](const Eigen::Matrix3d& e) {
    size_t support = 0;
    double cost = 0;
    for (size_t i = 0; i < ra.size(); ++i) {
      const double error = EpipolarSquared(e, ra[i], rb[i]);
      support += error <= 9;
      cost += std::min(error, 9.);
    }
    if (support > best || (support == best && cost < best_cost)) {
      best = support;
      best_cost = cost;
      essential = e;
      return true;
    }
    return false;
  };
  const uint32_t seed = matches.a * 73856093u ^ matches.b * 19349663u;
  const uint32_t usac_seed = (seed ^ 0x9e3779b9u) & 0x7fffffffu;
  cv::UsacParams usac;
  usac.confidence = .999;
  usac.maxIterations = 1000;
  usac.threshold = 3 / focal;
  usac.sampler = cv::SAMPLING_UNIFORM;
  usac.score = cv::SCORE_METHOD_MAGSAC;
  usac.loMethod = cv::LOCAL_OPTIM_SIGMA;
  usac.loSampleSize = 50;
  usac.loIterations = 10;
  usac.neighborsSearch = cv::NEIGH_FLANN_KNN;
  usac.final_polisher = cv::MAGSAC;
  usac.isParallel = false;
  usac.randomGeneratorState = int(usac_seed ? usac_seed : 1u);
  const cv::Mat eye = cv::Mat::eye(3, 3, CV_64F);
  const cv::Mat initial =
      cv::findEssentialMat(na, nb, eye, eye, cv::noArray(), cv::noArray(), cv::noArray(), usac);
  for (int row = 0; row + 3 <= initial.rows; row += 3)
    score(Matrix(initial.rowRange(row, row + 3)));
  // Five-point hypotheses use perspective coordinates only for the minimal
  // solve. Consensus and local refinement use the native fisheye pixel metric.
  std::mt19937 rng(seed);
  std::uniform_int_distribution<size_t> choose(0, ra.size() - 1);
  size_t trials = 1024;
  for (size_t trial = 0; trial < trials; ++trial) {
    std::array<size_t, 5> ids;
    std::vector<cv::Point2d> sa, sb;
    for (size_t k = 0; k < ids.size(); ++k) {
      size_t id;
      do {
        id = choose(rng);
      } while (std::find(ids.begin(), ids.begin() + k, id) != ids.begin() + k);
      ids[k] = id;
      sa.push_back(na[id]);
      sb.push_back(nb[id]);
    }
    for (const auto& e : FivePoint(sa, sb))
      score(e);
    if (best >= 24) {
      const double success = std::pow(double(best) / ra.size(), 5);
      const double needed = success >= 1 ? 32 : std::ceil(std::log(.001) / std::log1p(-success));
      trials = std::min(trials, size_t(std::clamp(needed, 32., 1024.)));
    }
  }
  if (best < 24)
    return result;
  for (int round = 0; round < 3; ++round) {
    Eigen::Matrix3d candidate;
    if (!RefitEssential(ra, rb, essential, candidate) || !score(candidate))
      break;
  }
  cv::Mat E(3, 3, CV_64F), mask(int(ra.size()), 1, CV_8UC1);
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      E.at<double>(r, c) = essential(r, c);
  for (size_t i = 0; i < ra.size(); ++i)
    mask.at<uint8_t>(int(i)) = EpipolarSquared(essential, ra[i], rb[i]) <= 9 ? 255 : 0;
  cv::Mat rotation, translation;
  cv::recoverPose(E, na, nb, cv::Mat::eye(3, 3, CV_64F), rotation, translation, 10000, mask);
  if (rotation.empty())
    return result;
  result.relative.Set(
      Eigen::Quaterniond(Matrix(rotation)),
      {translation.at<double>(0), translation.at<double>(1), translation.at<double>(2)});
  const auto R = result.relative.Rotation();
  std::vector<double> angles;
  Pose origin;
  for (size_t i = 0; i < matches.matches.size(); ++i) {
    if (!mask.at<uint8_t>(int(i)))
      continue;
    const auto& m = matches.matches[i];
    const auto& xa = model.images[matches.a].pixels[m.a];
    const auto& xb = model.images[matches.b].pixels[m.b];
    Eigen::Vector3d point;
    if (Triangulate(origin, result.relative, ra[i].ray, rb[i].ray, point) &&
        (ca.Project(point) - xa).norm() <= 6 &&
        (cb.Project(result.relative.Apply(point)) - xb).norm() <= 6) {
      result.inliers.push_back(m);
      angles.push_back(std::acos(std::clamp(ra[i].ray.dot(R.conjugate() * rb[i].ray), -1., 1.)) *
                       180 / Pi);
    }
  }
  result.angle = angles.empty() ? 0 : Quantile(angles, .5);
  if (result.inliers.size() < 24)
    result.inliers.clear();
  return result;
}
struct ImuPriorContext {
  bool available = false;
  std::vector<Eigen::Quaterniond> body_rotation;
  std::array<Eigen::Matrix3d, 2> boresight{Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Identity()};
  std::array<bool, 2> verified{false, false};
  std::array<double, 2> boresight_median_deg{-1., -1.};
  std::array<std::string, 2> source{"none", "none"};
  size_t pairs_checked = 0, pairs_rescued = 0, pairs_overruled = 0, pairs_rejected = 0;
  size_t pnp_rescued = 0, pnp_overruled = 0, pnp_rejected = 0;

  void Initialize(const Scene& scene) {
    *this = ImuPriorContext{};
    auto kinematics = ReadExposureKinematics(scene);
    if (!kinematics.has_attitude || kinematics.body_rotation.size() != scene.images.size())
      return;
    for (const auto& q : kinematics.body_rotation)
      if (!q.coeffs().allFinite() || q.squaredNorm() <= 1e-12)
        return;
    body_rotation = std::move(kinematics.body_rotation);
    available = true;
  }

  bool RelativeRotation(const Scene& scene, uint32_t a, uint32_t b,
                        Eigen::Matrix3d& R_ba) const {
    if (!available)
      return false;
    const uint32_t ca = scene.images[a].camera, cb = scene.images[b].camera;
    if (!verified[ca] || !verified[cb])
      return false;
    const Eigen::Matrix3d R_body_ba =
        (body_rotation[b].conjugate() * body_rotation[a]).normalized().toRotationMatrix();
    R_ba = boresight[cb] * R_body_ba * boresight[ca].transpose();
    return true;
  }

  void CalibrateHandEye(const Scene& scene, const std::vector<Pair>& pairs) {
    if (!available)
      return;
    struct Obs {
      Eigen::Matrix3d A, B;
      Eigen::Matrix<double, 9, 9> rows;
      double w = 1.0;
    };
    for (uint32_t cid = 0; cid < 2; ++cid) {
      std::vector<Obs> obs;
      for (const auto& p : pairs) {
        if (p.inliers.size() < 48 || scene.images[p.a].camera != cid ||
            scene.images[p.b].camera != cid)
          continue;
        const Eigen::Matrix3d B =
            (body_rotation[p.b].conjugate() * body_rotation[p.a]).normalized().toRotationMatrix();
        if (RotationAngleDeg(B) < 2.0)
          continue;
        const Eigen::Matrix3d A = p.relative.Rotation().toRotationMatrix();
        Eigen::Matrix<double, 9, 9> rows = Eigen::Matrix<double, 9, 9>::Zero();
        for (int r = 0; r < 3; ++r)
          for (int c = 0; c < 3; ++c) {
            const int row_idx = 3 * r + c;
            for (int m = 0; m < 3; ++m)
              rows(row_idx, 3 * m + c) += A(r, m);
            for (int n = 0; n < 3; ++n)
              rows(row_idx, 3 * r + n) -= B(n, c);
          }
        obs.push_back({A, B, rows, 1.0});
      }
      if (obs.size() < 8)
        continue;
      Eigen::Matrix3d X = Eigen::Matrix3d::Identity();
      Eigen::Matrix<double, 9, 1> eigvals = Eigen::Matrix<double, 9, 1>::Zero();
      size_t inliers = 0;
      std::vector<double> errors_deg;
      for (int pass = 0; pass < 3; ++pass) {
        Eigen::Matrix<double, 9, 9> N = Eigen::Matrix<double, 9, 9>::Zero();
        double sum_w = 0;
        for (const auto& item : obs)
          if (item.w > 0) {
            N.noalias() += item.w * item.rows.transpose() * item.rows;
            sum_w += item.w;
          }
        if (sum_w <= 0) {
          inliers = 0;
          break;
        }
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> solver(N);
        if (solver.info() != Eigen::Success) {
          inliers = 0;
          break;
        }
        eigvals = solver.eigenvalues();
        const Eigen::Matrix<double, 9, 1> v = solver.eigenvectors().col(0);
        for (int r = 0; r < 3; ++r)
          for (int c = 0; c < 3; ++c)
            X(r, c) = v(3 * r + c);
        if (X.determinant() < 0)
          X = -X;
        const auto svd = X.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        if ((U * svd.matrixV().transpose()).determinant() < 0)
          U.col(2) = -U.col(2);
        X = U * svd.matrixV().transpose();
        if (!X.allFinite()) {
          inliers = 0;
          break;
        }
        inliers = 0;
        errors_deg.clear();
        errors_deg.reserve(obs.size());
        for (auto& item : obs) {
          const double theta =
              RotationAngleDeg(item.A.transpose() * X * item.B * X.transpose());
          errors_deg.push_back(theta);
          if (theta <= 3.0) {
            item.w = 1.0;
            ++inliers;
          } else if (theta <= 10.0) {
            const double s = 3.0 / theta;
            item.w = s * s;
          } else {
            item.w = 0.0;
          }
        }
      }
      if (inliers < 8 || errors_deg.size() != obs.size())
        continue;
      if (eigvals[1] / std::max(eigvals[0], 1e-12) < 2.0)
        continue;
      const double median = Quantile(errors_deg, .5);
      if (median <= 2.0) {
        boresight[cid] = X;
        verified[cid] = true;
        boresight_median_deg[cid] = median;
        source[cid] = "hand_eye";
      }
    }
    Log("IMU hand-eye boresight: L=" +
        (verified[0] ? std::to_string(boresight_median_deg[0]) + " deg" : std::string("unverified")) +
        ", R=" +
        (verified[1] ? std::to_string(boresight_median_deg[1]) + " deg" : std::string("unverified")));
  }

  void RefreshFromNavigationModel(const Reconstruction& model) {
    if (!available)
      return;
    for (uint32_t cid = 0; cid < 2; ++cid) {
      std::vector<Eigen::Quaterniond> samples;
      for (uint32_t i = 0; i < model.images.size(); ++i)
        if (model.images[i].registered && model.images[i].camera == cid)
          samples.push_back((model.images[i].pose.Rotation() * body_rotation[i]).normalized());
      if (samples.size() < 8)
        continue;
      std::vector<uint8_t> inlier(samples.size(), 1);
      Eigen::Quaterniond q_mean = Eigen::Quaterniond::Identity();
      bool ok = true;
      for (int pass = 0; pass < 3; ++pass) {
        Eigen::Matrix4d sum = Eigen::Matrix4d::Zero();
        size_t count = 0;
        for (size_t k = 0; k < samples.size(); ++k)
          if (inlier[k]) {
            sum += samples[k].coeffs() * samples[k].coeffs().transpose();
            ++count;
          }
        if (count < 4) {
          ok = false;
          break;
        }
        const Eigen::Vector4d eig =
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d>(sum).eigenvectors().col(3);
        q_mean = Eigen::Quaterniond(eig[3], eig[0], eig[1], eig[2]).normalized();
        for (size_t k = 0; k < samples.size(); ++k)
          inlier[k] = samples[k].angularDistance(q_mean) <= 5.0 * Pi / 180.0;
      }
      if (!ok)
        continue;
      std::vector<double> errors_deg;
      errors_deg.reserve(samples.size());
      for (const auto& q : samples)
        errors_deg.push_back(q.angularDistance(q_mean) * 180.0 / Pi);
      const double median = Quantile(errors_deg, .5);
      if (median <= 2.0) {
        boresight[cid] = q_mean.toRotationMatrix();
        verified[cid] = true;
        boresight_median_deg[cid] = median;
        source[cid] = "navigation";
      }
    }
    Log("IMU navigation boresight: L=" +
        (verified[0] ? std::to_string(boresight_median_deg[0]) + " deg" : std::string("unverified")) +
        ", R=" +
        (verified[1] ? std::to_string(boresight_median_deg[1]) + " deg" : std::string("unverified")));
  }

  std::vector<RelativeRotationPrior> BuildRelativeRotationPriors(const Scene& scene) const {
    std::vector<RelativeRotationPrior> priors;
    if (!available)
      return priors;
    for (uint32_t cid = 0; cid < 2; ++cid) {
      if (!verified[cid])
        continue;
      std::vector<uint32_t> ids;
      for (uint32_t i = 0; i < scene.images.size(); ++i)
        if (scene.images[i].camera == cid)
          ids.push_back(i);
      std::sort(ids.begin(), ids.end(), [&](uint32_t a, uint32_t b) {
        return ExposureTime(scene.images[a]) < ExposureTime(scene.images[b]);
      });
      for (size_t k = 1; k < ids.size(); ++k) {
        const uint32_t a = ids[k - 1], b = ids[k];
        const double dt = ExposureTime(scene.images[b]) - ExposureTime(scene.images[a]);
        if (dt <= 0 || dt > 5.0)
          continue;
        Eigen::Matrix3d R_ba;
        if (RelativeRotation(scene, a, b, R_ba))
          priors.push_back(
              {a, b, Eigen::Quaterniond(R_ba).normalized(), ImuRelativeRotationSigma});
      }
    }
    return priors;
  }
};

bool RefineNearRotation(const std::vector<Measurement>& ra, const std::vector<Measurement>& rb,
                        const std::vector<size_t>& inliers, const Eigen::Matrix3d& R_prior,
                        double sigma_rad, double radius_px, Eigen::Matrix3d& R_inout,
                        Eigen::Vector3d& t_inout) {
  if (inliers.size() < 5 || t_inout.norm() < 1e-12)
    return false;
  const int n = int(inliers.size());
  const size_t m = size_t(n) + 3;
  const double inv_r = 1.0 / std::max(radius_px, 1e-12);
  const Eigen::AngleAxisd aa0(R_inout * R_prior.transpose());
  const Eigen::Vector3d d0 = aa0.angle() * aa0.axis();
  double x[5] = {d0.x(), d0.y(), d0.z(), 0.0, 0.0};
  Eigen::Vector3d t0 = t_inout.normalized();
  Eigen::Vector3d tu, tv;
  auto tangent_basis = [&]() {
    const Eigen::Vector3d ax =
        std::abs(t0.x()) < .9 ? Eigen::Vector3d::UnitX() : Eigen::Vector3d::UnitY();
    tu = ax.cross(t0).normalized();
    tv = t0.cross(tu);
  };
  tangent_basis();
  auto t_at = [&](const double* q) {
    return (t0 + q[3] * tu + q[4] * tv).normalized();
  };
  auto eval_at = [&](const double* q, std::vector<double>& e) {
    const Eigen::Matrix3d R = ExpSO3({q[0], q[1], q[2]}) * R_prior;
    const Eigen::Matrix3d E = CrossMatrix(t_at(q)) * R;
    e.resize(m);
    for (int k = 0; k < n; ++k)
      e[size_t(k)] = inv_r * EpipolarSigned(E, ra[inliers[size_t(k)]], rb[inliers[size_t(k)]]);
    for (int i = 0; i < 3; ++i)
      e[size_t(n) + i] = sigma_rad > 0 ? q[i] / sigma_rad : 0.0;
  };
  auto robust_cost = [&](const std::vector<double>& e) {
    double c = 0;
    for (int k = 0; k < n; ++k)
      c += std::log1p(e[size_t(k)] * e[size_t(k)]);
    for (int i = 0; i < 3; ++i)
      c += e[size_t(n) + i] * e[size_t(n) + i];
    return c;
  };
  std::vector<double> e0, ep, em, w(m, 1.0);
  std::vector<Eigen::Matrix<double, 5, 1>> J(m);
  eval_at(x, e0);
  double c0 = robust_cost(e0);
  double lambda = 1e-3;
  for (int it = 0; it < 15; ++it) {
    for (int k = 0; k < n; ++k)
      w[size_t(k)] = 1.0 / (1.0 + e0[size_t(k)] * e0[size_t(k)]);
    for (int p = 0; p < 5; ++p) {
      const double h = 1e-6;
      double xp[5], xm[5];
      std::copy_n(x, 5, xp);
      std::copy_n(x, 5, xm);
      xp[p] += h;
      xm[p] -= h;
      eval_at(xp, ep);
      eval_at(xm, em);
      for (size_t k = 0; k < m; ++k)
        J[k][p] = (ep[k] - em[k]) / (2.0 * h);
    }
    Eigen::Matrix<double, 5, 5> H = Eigen::Matrix<double, 5, 5>::Zero();
    Eigen::Matrix<double, 5, 1> g = Eigen::Matrix<double, 5, 1>::Zero();
    for (size_t k = 0; k < m; ++k) {
      H.noalias() += w[k] * J[k] * J[k].transpose();
      g.noalias() += w[k] * J[k] * e0[k];
    }
    bool accepted = false;
    for (int tries = 0; tries < 6 && !accepted; ++tries) {
      Eigen::Matrix<double, 5, 5> A = H;
      for (int a = 0; a < 5; ++a)
        A(a, a) += lambda * std::max(H(a, a), 1e-12);
      const Eigen::LDLT<Eigen::Matrix<double, 5, 5>> ldlt(A);
      if (ldlt.info() != Eigen::Success) {
        lambda *= 10.0;
        continue;
      }
      const Eigen::Matrix<double, 5, 1> d = ldlt.solve(-g);
      if (!d.allFinite()) {
        lambda *= 10.0;
        continue;
      }
      double xn[5];
      for (int k = 0; k < 5; ++k)
        xn[k] = x[k] + d[k];
      eval_at(xn, ep);
      const double c1 = robust_cost(ep);
      if (c1 < c0) {
        const double drop = (c0 - c1) / std::max(c0, 1e-300);
        for (int k = 0; k < 3; ++k)
          x[k] = xn[k];
        t0 = t_at(xn);
        x[3] = x[4] = 0.0;
        tangent_basis();
        e0.swap(ep);
        c0 = c1;
        lambda = std::max(lambda * 0.3, 1e-9);
        accepted = true;
        if (drop < 1e-6)
          it = 15;
      } else {
        lambda *= 10.0;
      }
    }
    if (!accepted)
      break;
  }
  const Eigen::Matrix3d R_out = ExpSO3({x[0], x[1], x[2]}) * R_prior;
  if (!R_out.allFinite() || !t0.allFinite())
    return false;
  R_inout = R_out;
  t_inout = t0;
  return true;
}

Pair VerifyKnownRotation(const Scene& scene, const Reconstruction& model, const MatchSet& matches,
                         const Eigen::Matrix3d& R_prior, const Pose* free_pose) {
  Pair result{matches.a, matches.b};
  if (matches.matches.size() < 24)
    return result;
  const auto& ca = model.cameras[scene.images[matches.a].camera];
  const auto& cb = model.cameras[scene.images[matches.b].camera];
  std::vector<Measurement> ra, rb;
  std::vector<Eigen::Vector3d> epipolar_normals;
  ra.reserve(matches.matches.size());
  rb.reserve(matches.matches.size());
  epipolar_normals.reserve(matches.matches.size());
  for (const auto& m : matches.matches) {
    const auto x = Measure(ca, model.images[matches.a].pixels.at(m.a));
    const auto y = Measure(cb, model.images[matches.b].pixels.at(m.b));
    ra.push_back(x);
    rb.push_back(y);
    epipolar_normals.push_back(y.ray.cross(R_prior * x.ray));
  }
  const double focal = (ca.params[0] + ca.params[1] + cb.params[0] + cb.params[1]) / 4;
  constexpr double rot_sigma_rad = 1.5 * Pi / 180.0;
  const double loose_px = std::min(30.0, 3.0 + rot_sigma_rad * focal);
  const double loose_sq = loose_px * loose_px;
  Eigen::Vector3d best_t = Eigen::Vector3d::Zero();
  size_t best_loose = 0;
  double best_loose_cost = INFINITY;
  auto score_t = [&](const Eigen::Vector3d& t) {
    const Eigen::Matrix3d E = CrossMatrix(t) * R_prior;
    size_t support = 0;
    double cost = 0;
    for (size_t i = 0; i < ra.size(); ++i) {
      const double err = EpipolarSquared(E, ra[i], rb[i]);
      support += err <= loose_sq;
      cost += std::min(err, loose_sq);
    }
    if (support > best_loose || (support == best_loose && cost < best_loose_cost)) {
      best_loose = support;
      best_loose_cost = cost;
      best_t = t;
    }
  };
  const uint32_t seed = matches.a * 73856093u ^ matches.b * 19349663u ^ 0x517cc1b7u;
  std::mt19937 rng(seed);
  std::uniform_int_distribution<size_t> choose(0, ra.size() - 1);
  size_t trials = 256;
  for (size_t trial = 0; trial < trials; ++trial) {
    const size_t i = choose(rng);
    size_t j = 0;
    do {
      j = choose(rng);
    } while (j == i);
    const Eigen::Vector3d t = epipolar_normals[i].cross(epipolar_normals[j]);
    const double norm = t.norm();
    if (norm < 1e-10)
      continue;
    score_t(t / norm);
    if (best_loose >= 24) {
      const double success = std::pow(double(best_loose) / ra.size(), 2);
      const double needed = success >= 1 ? 32 : std::ceil(std::log(.001) / std::log1p(-success));
      trials = std::min(trials, size_t(std::clamp(needed, 32., 256.)));
    }
  }
  if (best_loose < 24)
    return result;
  std::vector<size_t> loose_inliers;
  const Eigen::Matrix3d E_loose = CrossMatrix(best_t) * R_prior;
  for (size_t i = 0; i < ra.size(); ++i)
    if (EpipolarSquared(E_loose, ra[i], rb[i]) <= loose_sq)
      loose_inliers.push_back(i);

  auto eval_strict = [&](const Eigen::Matrix3d& R_cand, const Eigen::Vector3d& t_cand,
                         size_t& support, double& cost) {
    const Eigen::Matrix3d E = CrossMatrix(t_cand) * R_cand;
    support = 0;
    cost = 0;
    for (size_t i = 0; i < ra.size(); ++i) {
      const double err = EpipolarSquared(E, ra[i], rb[i]);
      support += err <= 9.0;
      cost += std::min(err, 9.0);
    }
  };

  Eigen::Matrix3d R_best = R_prior;
  Eigen::Vector3d t_best = best_t;
  bool have_refined =
      RefineNearRotation(ra, rb, loose_inliers, R_prior, rot_sigma_rad, 3.0, R_best, t_best) &&
      RotationAngleDeg(R_best * R_prior.transpose()) <= 5.0;
  size_t best_strict = 0;
  double best_strict_cost = INFINITY;
  if (have_refined)
    eval_strict(R_best, t_best, best_strict, best_strict_cost);
  if (free_pose &&
      RotationAngleDeg(free_pose->Rotation().toRotationMatrix() * R_prior.transpose()) <= 10.0) {
    Eigen::Matrix3d R_alt = free_pose->Rotation().toRotationMatrix();
    Eigen::Vector3d t_alt = free_pose->Translation();
    if (t_alt.norm() >= 1e-12) {
      t_alt.normalize();
      if (RefineNearRotation(ra, rb, loose_inliers, R_prior, rot_sigma_rad, 3.0, R_alt, t_alt) &&
          RotationAngleDeg(R_alt * R_prior.transpose()) <= 5.0) {
        size_t alt_strict = 0;
        double alt_cost = INFINITY;
        eval_strict(R_alt, t_alt, alt_strict, alt_cost);
        if (!have_refined || alt_strict > best_strict ||
            (alt_strict == best_strict && alt_cost < best_strict_cost)) {
          have_refined = true;
          R_best = R_alt;
          t_best = t_alt;
          best_strict = alt_strict;
          best_strict_cost = alt_cost;
        }
      }
    }
  }
  if (!have_refined || best_strict < 24)
    return result;

  const Eigen::Matrix3d E_final = CrossMatrix(t_best) * R_best;
  std::vector<uint8_t> strict_mask(ra.size(), 0);
  for (size_t i = 0; i < ra.size(); ++i)
    strict_mask[i] = EpipolarSquared(E_final, ra[i], rb[i]) <= 9.0 ? 1 : 0;

  const Pose origin;
  const Eigen::Quaterniond q_best(R_best);
  std::vector<Match> best_inliers;
  std::vector<double> best_angles;
  Pose best_pose;
  for (double sign : {1.0, -1.0}) {
    Pose cand;
    cand.Set(q_best, sign * t_best);
    std::vector<Match> inliers;
    std::vector<double> angles;
    for (size_t i = 0; i < matches.matches.size(); ++i) {
      if (!strict_mask[i])
        continue;
      const auto& m = matches.matches[i];
      const auto& xa = model.images[matches.a].pixels[m.a];
      const auto& xb = model.images[matches.b].pixels[m.b];
      Eigen::Vector3d point;
      if (Triangulate(origin, cand, ra[i].ray, rb[i].ray, point) &&
          (ca.Project(point) - xa).norm() <= 6 &&
          (cb.Project(cand.Apply(point)) - xb).norm() <= 6) {
        inliers.push_back(m);
        angles.push_back(
            std::acos(std::clamp(ra[i].ray.dot(q_best.conjugate() * rb[i].ray), -1., 1.)) *
            180.0 / Pi);
      }
    }
    if (inliers.size() > best_inliers.size()) {
      best_pose = cand;
      best_inliers = std::move(inliers);
      best_angles = std::move(angles);
    }
  }
  if (best_inliers.size() < 24)
    return result;
  result.relative = best_pose;
  result.inliers = std::move(best_inliers);
  result.angle = best_angles.empty() ? 0 : Quantile(best_angles, .5);
  return result;
}

// Each pair is verified independently with its own fixed seed, so the result does not depend on
// the thread count.
std::vector<Pair> VerifyPairs(const Scene& scene, const Reconstruction& model, const fs::path& work,
                              const std::vector<std::array<uint32_t, 2>>& candidates, int workers,
                              ImuPriorContext& imu) {
  std::vector<Pair> pairs(candidates.size());
  if (candidates.empty())
    return pairs;
  const auto archive_path = MatchesPath(work);
  const auto index = IndexMatchArchive(archive_path, false);
  std::atomic<size_t> next{0}, done{0};
  const int threads = std::min<int>(workers, int(candidates.size()));
  ParallelFor(size_t(threads), threads, 1, [&](size_t) {
    std::ifstream stream(archive_path, std::ios::binary);
    if (!stream)
      throw std::runtime_error("Cannot read match archive: " + archive_path.u8string());
    for (;;) {
      const size_t i = next.fetch_add(1);
      if (i >= candidates.size())
        break;
      const auto p = candidates[i];
      const auto it = index.find({p[0], p[1]});
      if (it == index.end())
        throw std::runtime_error("Missing match record in archive: " + std::to_string(p[0]) +
                                 "_" + std::to_string(p[1]));
      pairs[i] = Verify(scene, model, ReadMatchRecord(stream, it->second, p[0], p[1]));
      const auto n = done.fetch_add(1) + 1;
      if (n % 500 == 0)
        Log("Geometry " + std::to_string(n) + "/" + std::to_string(candidates.size()));
    }
  });
  imu.CalibrateHandEye(scene, pairs);
  if (imu.verified[0] || imu.verified[1]) {
    std::vector<size_t> check_indices;
    std::vector<Eigen::Matrix3d> check_priors;
    std::vector<double> check_angles;
    for (size_t i = 0; i < pairs.size(); ++i) {
      Eigen::Matrix3d R_prior;
      if (!imu.RelativeRotation(scene, candidates[i][0], candidates[i][1], R_prior))
        continue;
      if (pairs[i].inliers.empty()) {
        check_indices.push_back(i);
        check_priors.push_back(R_prior);
        check_angles.push_back(0.0);
      } else {
        const double angle_err =
            RotationAngleDeg(pairs[i].relative.Rotation().toRotationMatrix() * R_prior.transpose());
        if (angle_err > 3.0) {
          check_indices.push_back(i);
          check_priors.push_back(R_prior);
          check_angles.push_back(angle_err);
        }
      }
    }
    imu.pairs_checked = check_indices.size();
    if (!check_indices.empty()) {
      enum class Outcome : uint8_t { Unchanged = 0, Rescued = 1, Overruled = 2, Rejected = 3 };
      std::vector<Outcome> outcomes(check_indices.size(), Outcome::Unchanged);
      std::atomic<size_t> next_check{0};
      const int check_threads = std::min<int>(workers, int(check_indices.size()));
      ParallelFor(size_t(check_threads), check_threads, 1, [&](size_t) {
        std::ifstream stream(archive_path, std::ios::binary);
        if (!stream)
          throw std::runtime_error("Cannot read match archive: " + archive_path.u8string());
        for (;;) {
          const size_t k = next_check.fetch_add(1);
          if (k >= check_indices.size())
            break;
          const size_t idx = check_indices[k];
          const auto p = candidates[idx];
          const auto it = index.find({p[0], p[1]});
          if (it == index.end())
            throw std::runtime_error("Missing match record in archive: " + std::to_string(p[0]) +
                                     "_" + std::to_string(p[1]));
          const auto matches = ReadMatchRecord(stream, it->second, p[0], p[1]);
          if (pairs[idx].inliers.empty()) {
            Pair known = VerifyKnownRotation(scene, model, matches, check_priors[k], nullptr);
            if (!known.inliers.empty()) {
              pairs[idx] = std::move(known);
              outcomes[k] = Outcome::Rescued;
            }
          } else {
            Pair known = VerifyKnownRotation(scene, model, matches, check_priors[k],
                                             &pairs[idx].relative);
            const double angle_err = check_angles[k];
            if (!known.inliers.empty() &&
                (known.inliers.size() * 10 >= pairs[idx].inliers.size() * 7 || angle_err > 6.0)) {
              pairs[idx] = std::move(known);
              outcomes[k] = Outcome::Overruled;
            } else if (known.inliers.empty() && angle_err > 6.0) {
              pairs[idx].inliers.clear();
              outcomes[k] = Outcome::Rejected;
            }
          }
        }
      });
      for (Outcome o : outcomes) {
        if (o == Outcome::Rescued)
          ++imu.pairs_rescued;
        else if (o == Outcome::Overruled)
          ++imu.pairs_overruled;
        else if (o == Outcome::Rejected)
          ++imu.pairs_rejected;
      }
      Log("IMU verification pass: checked " + std::to_string(imu.pairs_checked) + ", rescued " +
          std::to_string(imu.pairs_rescued) + ", overruled " +
          std::to_string(imu.pairs_overruled) + ", rejected " +
          std::to_string(imu.pairs_rejected));
    }
  }
  return pairs;
}
using FeatureColors = std::vector<std::vector<std::array<uint8_t, 3>>>;
// Cached features become the observations of a new model, or must match those of a resumed one.
FeatureColors LoadFeatures(const Scene& scene, const fs::path& work, Reconstruction& m,
                           bool resumed) {
  FeatureColors colors(m.images.size());
  for (uint32_t id = 0; id < m.images.size(); ++id) {
    auto f = ReadFeatures(FeaturePath(work, id));
    if (f.image != id)
      throw std::runtime_error("Feature/image identity differs");
    if (resumed) {
      if (m.images[id].camera != scene.images[id].camera ||
          m.images[id].pixels.size() != f.pixels.size())
        throw std::runtime_error("Incremental feature identity differs");
      for (size_t k = 0; k < f.pixels.size(); ++k)
        if (m.images[id].pixels[k] != Eigen::Vector2d(f.pixels[k][0], f.pixels[k][1]))
          throw std::runtime_error("Incremental feature coordinates differ");
    } else {
      m.images[id].camera = scene.images[id].camera;
      for (const auto& p : f.pixels)
        m.images[id].pixels.emplace_back(p[0], p[1]);
      m.images[id].points.assign(f.pixels.size(), NoPoint);
    }
    colors[id] = std::move(f.colors);
  }
  return colors;
}
// Each point takes the colour of its feature in the first image of its track.
void ColorPoints(Reconstruction& m, const FeatureColors& colors) {
  for (uint32_t id = 0; id < m.images.size(); ++id) {
    for (uint32_t k = 0; k < m.images[id].points.size(); ++k)
      if (m.images[id].points[k] != NoPoint) {
        auto& p = m.points[m.images[id].points[k]];
        if (p.track.front().image != id)
          continue;
        p.color = colors[id][k];
      }
  }
}
struct Tracks {
  std::vector<std::vector<Observation>> tracks;
  std::vector<std::vector<uint32_t>> lookup;
  explicit Tracks(const Reconstruction& model, const std::vector<Pair>& pairs) {
    std::vector<size_t> offset{0};
    for (const auto& image : model.images)
      offset.push_back(offset.back() + image.pixels.size());
    std::vector<size_t> parent(offset.back()), sizes(offset.back(), 1);
    std::iota(parent.begin(), parent.end(), 0);
    std::unordered_map<size_t, std::vector<uint32_t>> labels;
    auto root = [&](size_t x) {
      while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    };
    auto images = [&](size_t r) {
      const auto it = labels.find(r);
      return it != labels.end()
                 ? it->second
                 : std::vector<uint32_t>{uint32_t(
                       std::upper_bound(offset.begin(), offset.end(), r) - offset.begin() - 1)};
    };
    for (const auto& pair : pairs)
      for (const auto& match : pair.inliers) {
        auto a = root(offset[pair.a] + match.a), b = root(offset[pair.b] + match.b);
        if (a == b)
          continue;
        auto ia = images(a), ib = images(b);
        bool conflict = false;
        for (uint32_t id : ia)
          if (std::binary_search(ib.begin(), ib.end(), id)) {
            conflict = true;
            break;
          }
        if (conflict)
          continue;
        if (sizes[a] < sizes[b]) {
          std::swap(a, b);
          std::swap(ia, ib);
        }
        std::vector<uint32_t> merged;
        std::merge(ia.begin(), ia.end(), ib.begin(), ib.end(), std::back_inserter(merged));
        parent[b] = a;
        sizes[a] += sizes[b];
        labels[a] = std::move(merged);
        labels.erase(b);
      }
    lookup.resize(model.images.size());
    std::unordered_map<size_t, uint32_t> ids;
    for (uint32_t im = 0; im < model.images.size(); ++im) {
      lookup[im].assign(model.images[im].pixels.size(), UINT32_MAX);
      for (uint32_t k = 0; k < model.images[im].pixels.size(); ++k) {
        const auto r = root(offset[im] + k);
        if (sizes[r] < 2)
          continue;
        auto it = ids.find(r);
        if (it == ids.end()) {
          const uint32_t id = uint32_t(tracks.size());
          tracks.emplace_back();
          it = ids.emplace(r, id).first;
        }
        lookup[im][k] = it->second;
        tracks[it->second].push_back({im, k});
      }
    }
  }
};
bool AddPoint(Reconstruction& model, const std::vector<Observation>& observations,
              uint64_t& identity, bool refine = false) {
  if (identity != NoPoint && !model.points[identity].track.empty()) {
    auto& point = model.points[identity];
    std::set<uint32_t> seen;
    for (const auto& o : point.track)
      seen.insert(o.image);
    const size_t before = point.track.size();
    for (const auto& o : observations) {
      const auto& image = model.images[o.image];
      if (!image.registered || seen.count(o.image) ||
          (image.points[o.feature] != NoPoint && image.points[o.feature] != identity))
        continue;
      const auto v = image.pose.Apply(point.xyz);
      if (v.z() > 0 &&
          (model.cameras[image.camera].Project(v) - image.pixels[o.feature]).norm() <= 4.5) {
        point.track.push_back(o);
        model.images[o.image].points[o.feature] = identity;
        seen.insert(o.image);
      }
    }
    if (refine || point.track.size() > before)
      RefinePoint(model, point.track, point.xyz);
    return true;
  }
  std::vector<Observation> track;
  for (const auto& o : observations)
    if (model.images[o.image].registered && (model.images[o.image].points[o.feature] == NoPoint ||
                                             model.images[o.image].points[o.feature] == identity))
      track.push_back(o);
  if (track.size() < 2)
    return false;
  double best = 1;
  size_t first = 0, second = 0;
  std::vector<Eigen::Vector3d> rays, world;
  for (const auto& o : track) {
    const auto& image = model.images[o.image];
    const auto ray = model.cameras[image.camera].Ray(image.pixels[o.feature]);
    rays.push_back(ray);
    world.push_back(image.pose.Rotation().conjugate() * ray);
  }
  for (size_t i = 0; i < track.size(); ++i)
    for (size_t j = i + 1; j < track.size(); ++j) {
      const double cosine = world[i].dot(world[j]);
      if (cosine < best) {
        best = cosine;
        first = i;
        second = j;
      }
    }
  if (best > std::cos(.5 * Pi / 180))
    return false;
  Eigen::Vector3d point;
  if (!Triangulate(model.images[track[first].image].pose, model.images[track[second].image].pose,
                   rays[first], rays[second], point))
    return false;
  std::vector<Observation> good;
  for (const auto& o : track) {
    const auto& im = model.images[o.image];
    const auto v = im.pose.Apply(point);
    if (v.z() > 0 && (model.cameras[im.camera].Project(v) - im.pixels[o.feature]).norm() <= 6)
      good.push_back(o);
  }
  if (good.size() < 2)
    return false;
  RefinePoint(model, good, point);
  if (identity == NoPoint) {
    identity = model.points.size();
    model.points.push_back({point, {127, 127, 127}, 0, good});
  } else {
    model.points[identity].xyz = point;
    model.points[identity].track = std::move(good);
  }
  for (const auto& o : model.points[identity].track)
    model.images[o.image].points[o.feature] = identity;
  return true;
}
Json CompleteTracks(Reconstruction& model, const Tracks& graph, std::vector<uint64_t>& identities) {
  const auto started = Clock::now();
  const size_t before = model.Observations();
  size_t merged = 0;
  for (size_t i = 0; i < graph.tracks.size(); ++i) {
    std::set<uint64_t> existing;
    for (const auto& o : graph.tracks[i])
      if (model.images[o.image].points[o.feature] != NoPoint)
        existing.insert(model.images[o.image].points[o.feature]);
    if (existing.size() == 1)
      identities[i] = *existing.begin();
    else if (existing.size() > 1) {
      std::map<uint32_t, uint32_t> features;
      std::vector<Observation> track;
      bool conflict = false;
      uint64_t winner = *existing.begin();
      for (uint64_t id : existing) {
        if (model.points[id].track.size() > model.points[winner].track.size())
          winner = id;
        for (const auto& o : model.points[id].track) {
          const auto inserted = features.emplace(o.image, o.feature);
          if (!inserted.second && inserted.first->second != o.feature)
            conflict = true;
          if (inserted.second)
            track.push_back(o);
        }
      }
      if (conflict)
        continue;
      Eigen::Vector3d point = model.points[winner].xyz;
      RefinePoint(model, track, point);
      bool valid = true;
      for (const auto& o : track) {
        const auto& image = model.images[o.image];
        const auto v = image.pose.Apply(point);
        valid = valid && v.z() > 0 &&
                (model.cameras[image.camera].Project(v) - image.pixels[o.feature]).norm() <= 4.5;
      }
      if (!valid)
        continue;
      for (uint64_t id : existing)
        if (id != winner) {
          model.points[id].track.clear();
          ++merged;
        }
      model.points[winner].xyz = point;
      model.points[winner].track = std::move(track);
      for (const auto& o : model.points[winner].track)
        model.images[o.image].points[o.feature] = winner;
      identities[i] = winner;
    }
    AddPoint(model, graph.tracks[i], identities[i], true);
  }
  const size_t completed = model.Observations();
  model.Filter();
  return {{"added_observations", completed - before},
          {"removed_observations", completed - model.Observations()},
          {"merged_points", merged},
          {"seconds", Seconds(started)}};
}
Json CompleteUnassignedTracks(Reconstruction& model, const Tracks& graph) {
  const auto started = Clock::now();
  constexpr int maximum_rounds = 3;
  const size_t before_points = std::count_if(model.points.begin(), model.points.end(),
                                           [](const auto& p) { return !p.track.empty(); });
  const size_t before_observations = model.Observations();
  // A matching component can contain more than one landmark. Once its primary
  // point is established, observations rejected by that point must still get
  // a chance to form independent tracks. Keep the current cameras fixed.
  for (int round = 0; round < maximum_rounds; ++round) {
    const size_t before = model.Observations();
    for (const auto& track : graph.tracks) {
      std::vector<Observation> unassigned;
      for (const auto& o : track)
        if (model.images[o.image].registered && model.images[o.image].points[o.feature] == NoPoint)
          unassigned.push_back(o);
      if (unassigned.size() < 2)
        continue;
      uint64_t identity = NoPoint;
      AddPoint(model, unassigned, identity);
    }
    model.Filter();
    if (model.Observations() == before)
      break;
  }
  const size_t after_points = std::count_if(model.points.begin(), model.points.end(),
                                          [](const auto& p) { return !p.track.empty(); });
  return {{"added_points", after_points - before_points},
          {"added_observations", model.Observations() - before_observations},
          {"cameras_fixed", true}, {"seconds", Seconds(started)}};
}
size_t Correspondences(const Reconstruction& model, const std::vector<Pair>& pairs,
                       const std::vector<size_t>& neighbors, uint32_t id,
                       std::vector<cv::Point3d>* xyz = nullptr,
                       std::vector<cv::Point2d>* xy = nullptr,
                       std::vector<uint32_t>* features = nullptr,
                       std::vector<uint64_t>* identities = nullptr) {
  const auto& im = model.images[id];
  const auto& cam = model.cameras[im.camera];
  std::vector<uint8_t> supported(im.pixels.size());
  struct Vote {
    size_t count = 0;
    double score = 0;
  };
  std::map<uint32_t, std::map<uint64_t, Vote>> votes;
  for (size_t index : neighbors) {
    const auto& pair = pairs[index];
    const bool first = pair.a == id;
    const auto& other = model.images[first ? pair.b : pair.a];
    if (!other.registered)
      continue;
    for (const auto& match : pair.inliers) {
      const uint32_t feature = first ? match.a : match.b;
      const uint64_t point = other.points[first ? match.b : match.a];
      if (point == NoPoint)
        continue;
      supported[feature] = 1;
      if (xyz) {
        auto& vote = votes[feature][point];
        ++vote.count;
        vote.score += match.score;
      }
    }
  }
  if (!xyz)
    return std::count(supported.begin(), supported.end(), uint8_t(1));
  struct Candidate {
    uint32_t feature;
    uint64_t point;
    Vote vote;
  };
  std::vector<Candidate> candidates;
  for (const auto& [feature, options] : votes) {
    Candidate best{feature, NoPoint, {}};
    for (const auto& [point, vote] : options)
      if (vote.count > best.vote.count ||
          (vote.count == best.vote.count && vote.score > best.vote.score))
        best = {feature, point, vote};
    candidates.push_back(best);
  }
  std::sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) {
    if (a.vote.count != b.vote.count)
      return a.vote.count > b.vote.count;
    if (a.vote.score != b.vote.score)
      return a.vote.score > b.vote.score;
    return a.feature < b.feature;
  });
  std::set<uint64_t> used;
  for (const auto& candidate : candidates) {
    if (used.insert(candidate.point).second) {
      const auto& p = model.points[candidate.point].xyz;
      xyz->emplace_back(p.x(), p.y(), p.z());
      const auto ray = cam.Ray(im.pixels[candidate.feature]);
      xy->emplace_back(ray.x() / ray.z(), ray.y() / ray.z());
      features->push_back(candidate.feature);
      identities->push_back(candidate.point);
    }
  }
  return xyz->size();
}
bool PredictImageRotation(const Scene& scene, const Reconstruction& model,
                          const std::vector<Pair>& pairs, const std::vector<size_t>& neighbors,
                          uint32_t id, const ImuPriorContext& imu, bool in_nav_frame,
                          Eigen::Matrix3d& R_pred) {
  if (!imu.available)
    return false;
  size_t best_shared = 0, best_inliers = 0;
  uint32_t best_nbr = UINT32_MAX;
  Eigen::Matrix3d best_rel = Eigen::Matrix3d::Identity();
  for (size_t index : neighbors) {
    const auto& pair = pairs[index];
    const bool first = pair.a == id;
    const uint32_t nbr = first ? pair.b : pair.a;
    if (!model.images[nbr].registered)
      continue;
    Eigen::Matrix3d R_rel;
    if (!imu.RelativeRotation(scene, nbr, id, R_rel))
      continue;
    size_t shared = 0;
    for (const auto& m : pair.inliers)
      if (model.images[nbr].points[first ? m.b : m.a] != NoPoint)
        ++shared;
    if (shared == 0)
      continue;
    if (shared > best_shared || (shared == best_shared && pair.inliers.size() > best_inliers) ||
        (shared == best_shared && pair.inliers.size() == best_inliers && nbr < best_nbr)) {
      best_shared = shared;
      best_inliers = pair.inliers.size();
      best_nbr = nbr;
      best_rel = R_rel;
    }
  }
  if (best_nbr != UINT32_MAX) {
    R_pred = best_rel * model.images[best_nbr].pose.Rotation().toRotationMatrix();
    return true;
  }
  const uint32_t cam_id = scene.images[id].camera;
  if (in_nav_frame && imu.verified[cam_id] && imu.source[cam_id] == "navigation") {
    R_pred = imu.boresight[cam_id] * imu.body_rotation[id].conjugate().toRotationMatrix();
    return true;
  }
  return false;
}

bool SolvePnPKnownRotation(const Camera& cam, const ModelImage& image,
                           const std::vector<cv::Point3d>& xyz,
                           const std::vector<uint32_t>& features, const Eigen::Matrix3d& R_pred,
                           const Eigen::Vector3d* navigation, uint32_t id, Pose& out_pose) {
  const size_t n = xyz.size();
  if (n < 24)
    return false;
  std::vector<Eigen::Vector3d> pts(n), rays(n), rx(n);
  for (size_t i = 0; i < n; ++i) {
    pts[i] = {xyz[i].x, xyz[i].y, xyz[i].z};
    rays[i] = cam.Ray(image.pixels[features[i]]);
    rx[i] = R_pred * pts[i];
  }
  const Eigen::Quaterniond q_pred(R_pred);
  Pose best_pose;
  size_t best_loose = 0;
  double best_loose_cost = INFINITY;
  std::mt19937 rng(id * 2654435761u ^ 0x517cc1b7u);
  std::uniform_int_distribution<size_t> choose(0, n - 1);
  size_t trials = 512;
  for (size_t trial = 0; trial < trials; ++trial) {
    const size_t i = choose(rng);
    size_t j = 0;
    do {
      j = choose(rng);
    } while (j == i);
    if (std::abs(rays[i].dot(rays[j])) > 1. - 1e-8)
      continue;
    Eigen::Matrix3d N = Eigen::Matrix3d::Zero();
    Eigen::Vector3d rhs = Eigen::Vector3d::Zero();
    for (size_t k : {i, j}) {
      const Eigen::Matrix3d P = Eigen::Matrix3d::Identity() - rays[k] * rays[k].transpose();
      N.noalias() += P;
      rhs.noalias() -= P * rx[k];
    }
    const Eigen::LDLT<Eigen::Matrix3d> ldlt(N);
    if (ldlt.info() != Eigen::Success || ldlt.vectorD().minCoeff() <= 1e-8)
      continue;
    const Eigen::Vector3d t = ldlt.solve(rhs);
    if (!t.allFinite())
      continue;
    if ((rx[i] + t).dot(rays[i]) <= 0 || (rx[j] + t).dot(rays[j]) <= 0)
      continue;
    Pose cand;
    cand.Set(q_pred, t);
    if (navigation && (cand.Center() - *navigation).norm() > NavigationGate)
      continue;
    size_t support = 0;
    double cost = 0;
    for (size_t k = 0; k < n; ++k) {
      const Eigen::Vector3d v = cand.Apply(pts[k]);
      const double err =
          v.z() > 0 ? (cam.Project(v) - image.pixels[features[k]]).squaredNorm() : INFINITY;
      support += err <= 144.;
      cost += std::min(err, 144.);
    }
    if (support > best_loose || (support == best_loose && cost < best_loose_cost)) {
      best_pose = cand;
      best_loose = support;
      best_loose_cost = cost;
    }
    if (best_loose >= 24) {
      const double success = std::pow(double(best_loose) / n, 2);
      const double needed = success >= 1 ? 32 : std::ceil(std::log(.001) / std::log1p(-success));
      trials = std::min(trials, size_t(std::clamp(needed, 32., 512.)));
    }
  }
  if (best_loose < 24)
    return false;
  auto strict_score = [&](const Pose& pose, size_t& support, double& cost, size_t& cell_count) {
    if (navigation && (pose.Center() - *navigation).norm() > NavigationGate)
      return false;
    if (RotationAngleDeg(pose.Rotation().toRotationMatrix() * R_pred.transpose()) > 5.0)
      return false;
    support = 0;
    cost = 0;
    std::set<int> cells;
    for (size_t i = 0; i < n; ++i) {
      const Eigen::Vector3d v = pose.Apply(pts[i]);
      const Eigen::Vector2d& px = image.pixels[features[i]];
      const double err = v.z() > 0 ? (cam.Project(v) - px).squaredNorm() : INFINITY;
      if (err <= 36.) {
        ++support;
        cells.insert(cam.CoverageCell(px));
      }
      cost += std::min(err, 36.);
    }
    cell_count = cells.size();
    return true;
  };
  Pose cur = best_pose;
  size_t cur_support = 0, cur_cells = 0;
  double cur_cost = INFINITY;
  strict_score(cur, cur_support, cur_cost, cur_cells);
  for (int round = 0; round < 2; ++round) {
    const double threshold = round == 0 ? 12.0 : 6.0;
    std::vector<Eigen::Vector3d> fit_points;
    std::vector<Eigen::Vector2d> fit_pixels;
    for (size_t i = 0; i < n; ++i) {
      const Eigen::Vector3d v = cur.Apply(pts[i]);
      const Eigen::Vector2d& px = image.pixels[features[i]];
      if (v.z() > 0 && (cam.Project(v) - px).norm() <= threshold) {
        fit_points.push_back(pts[i]);
        fit_pixels.push_back(px);
      }
    }
    if (fit_points.size() < 6)
      break;
    Pose cand = cur;
    RefinePose(cam, cand, fit_points, fit_pixels);
    size_t cand_support = 0, cand_cells = 0;
    double cand_cost = INFINITY;
    if (strict_score(cand, cand_support, cand_cost, cand_cells) &&
        (cand_support > cur_support || (cand_support == cur_support && cand_cost < cur_cost))) {
      cur = cand;
      cur_support = cand_support;
      cur_cost = cand_cost;
      cur_cells = cand_cells;
    } else if (round == 1) {
      break;
    }
  }
  if (cur_support < MinimumImageObservations || cur_cells < MinimumImageCells)
    return false;
  out_pose = cur;
  return true;
}

// With the model in the navigation frame, pose hypotheses whose centre is beyond the navigation
// gate are rejected: repeated texture can otherwise support a pose shifted along the pattern.
bool Register(const Scene& scene, Reconstruction& model, const std::vector<Pair>& pairs,
              const std::vector<size_t>& neighbors, uint32_t id,
              const Eigen::Vector3d* navigation, ImuPriorContext& imu, bool in_nav_frame) {
  std::vector<cv::Point3d> xyz;
  std::vector<cv::Point2d> xy;
  std::vector<uint32_t> features;
  std::vector<uint64_t> identities;
  if (Correspondences(model, pairs, neighbors, id, &xyz, &xy, &features, &identities) < 24)
    return false;
  auto& image = model.images[id];
  const auto& cam = model.cameras[image.camera];
  Pose selected;
  size_t maximum = 0;
  double best_cost = INFINITY;
  auto score = [&](const Pose& pose) {
    if (navigation && (pose.Center() - *navigation).norm() > NavigationGate)
      return false;
    size_t support = 0;
    double cost = 0;
    for (size_t i = 0; i < xyz.size(); ++i) {
      const auto v = pose.Apply({xyz[i].x, xyz[i].y, xyz[i].z});
      const double error =
          v.z() > 0 ? (cam.Project(v) - image.pixels[features[i]]).squaredNorm() : INFINITY;
      support += error <= 36;
      cost += std::min(error, 36.);
    }
    if (support > maximum || (support == maximum && cost < best_cost)) {
      selected = pose;
      maximum = support;
      best_cost = cost;
      return true;
    }
    return false;
  };
  auto hypothesis = [&](const cv::Mat& rv, const cv::Mat& tv) {
    if (!cv::checkRange(rv) || !cv::checkRange(tv))
      return;
    cv::Mat R;
    cv::Rodrigues(rv, R);
    Pose pose;
    pose.Set(Eigen::Quaterniond(Matrix(R)), {tv.at<double>(0), tv.at<double>(1), tv.at<double>(2)});
    score(pose);
  };
  // Perspective coordinates are only used by the minimal solver. Every
  // consensus decision is measured in the original fisheye image pixels.
  std::mt19937 rng(id * 2654435761u ^ 0x9e3779b9u);
  std::uniform_int_distribution<size_t> choose(0, xyz.size() - 1);
  size_t trials = 2000;
  for (size_t trial = 0; trial < trials; ++trial) {
    std::array<size_t, 4> ids;
    std::vector<cv::Point3d> sample_xyz;
    std::vector<cv::Point2d> sample_xy;
    for (size_t k = 0; k < ids.size(); ++k) {
      size_t picked;
      do {
        picked = choose(rng);
      } while (std::find(ids.begin(), ids.begin() + k, picked) != ids.begin() + k);
      ids[k] = picked;
      sample_xyz.push_back(xyz[picked]);
      sample_xy.push_back(xy[picked]);
    }
    const auto ab = sample_xyz[1] - sample_xyz[0], ac = sample_xyz[2] - sample_xyz[0];
    if (cv::norm(ab.cross(ac)) <= 1e-8 * cv::norm(ab) * cv::norm(ac))
      continue;
    std::vector<cv::Mat> rotations, translations;
    cv::solvePnPGeneric(sample_xyz, sample_xy, cv::Mat::eye(3, 3, CV_64F), cv::noArray(), rotations,
                        translations, false, cv::SOLVEPNP_AP3P);
    for (size_t k = 0; k < rotations.size(); ++k)
      hypothesis(rotations[k], translations[k]);
    if (maximum >= 24) {
      const double success = std::pow(double(maximum) / xyz.size(), 4);
      const double needed = success >= 1 ? 32 : std::ceil(std::log(.001) / std::log1p(-success));
      trials = std::min(trials, size_t(std::clamp(needed, 32., 2000.)));
    }
  }
  bool free_ok = false;
  if (maximum >= 24) {
    for (int round = 0; round < 2; ++round) {
      std::vector<Eigen::Vector3d> fit_points;
      std::vector<Eigen::Vector2d> fit_pixels;
      for (size_t i = 0; i < xyz.size(); ++i) {
        const Eigen::Vector3d point(xyz[i].x, xyz[i].y, xyz[i].z);
        const auto v = selected.Apply(point);
        const auto& pixel = image.pixels[features[i]];
        if (v.z() > 0 && (cam.Project(v) - pixel).norm() <= 6) {
          fit_points.push_back(point);
          fit_pixels.push_back(pixel);
        }
      }
      Pose pose = selected;
      RefinePose(cam, pose, fit_points, fit_pixels);
      if (!score(pose))
        break;
    }
    std::set<int> cells;
    for (size_t i = 0; i < xyz.size(); ++i) {
      const auto v = selected.Apply({xyz[i].x, xyz[i].y, xyz[i].z});
      const auto& pixel = image.pixels[features[i]];
      if (v.z() > 0 && (cam.Project(v) - pixel).norm() <= 6)
        cells.insert(cam.CoverageCell(pixel));
    }
    free_ok = maximum >= MinimumImageObservations && cells.size() >= MinimumImageCells;
  }
  Eigen::Matrix3d R_pred;
  if (PredictImageRotation(scene, model, pairs, neighbors, id, imu, in_nav_frame, R_pred)) {
    if (!free_ok) {
      Pose known_pose;
      if (!SolvePnPKnownRotation(cam, image, xyz, features, R_pred, navigation, id, known_pose))
        return false;
      selected = known_pose;
      ++imu.pnp_rescued;
    } else {
      const double angle_err =
          RotationAngleDeg(selected.Rotation().toRotationMatrix() * R_pred.transpose());
      if (angle_err > 3.0) {
        Pose known_pose;
        if (SolvePnPKnownRotation(cam, image, xyz, features, R_pred, navigation, id, known_pose)) {
          selected = known_pose;
          ++imu.pnp_overruled;
        } else if (angle_err > 8.0) {
          ++imu.pnp_rejected;
          return false;
        }
      }
    }
  } else if (!free_ok) {
    return false;
  }
  image.pose = selected;
  image.registered = true;
  for (size_t i = 0; i < xyz.size(); ++i) {
    const auto v = selected.Apply(model.points[identities[i]].xyz);
    if (v.z() > 0 && (cam.Project(v) - image.pixels[features[i]]).norm() <= 6) {
      model.points[identities[i]].track.push_back({id, features[i]});
      image.points[features[i]] = identities[i];
    }
  }
  return true;
}
}  // namespace
std::set<uint64_t> CalibrationTracks(const Scene& scene, const Reconstruction& model,
                                     Json& report) {
  std::set<uint64_t> selected;
  std::vector<size_t> observations(model.images.size());
  std::vector<std::set<int>> cells(model.images.size());
  std::vector<uint32_t> parent(model.images.size());
  std::iota(parent.begin(), parent.end(), 0);
  auto root = [&](uint32_t id) {
    while (parent[id] != id) {
      parent[id] = parent[parent[id]];
      id = parent[id];
    }
    return id;
  };
  std::array<std::array<size_t, 4>, 2> angles{};
  for (uint64_t id = 0; id < model.points.size(); ++id) {
    const auto& point = model.points[id];
    if (point.track.size() < 3 || point.error > 2)
      continue;
    bool parallax = false;
    for (size_t a = 0; a < point.track.size() && !parallax; ++a)
      for (size_t b = a + 1; b < point.track.size(); ++b) {
        const auto ra = (point.xyz - model.images[point.track[a].image].pose.Center()).normalized();
        const auto rb = (point.xyz - model.images[point.track[b].image].pose.Center()).normalized();
        if (ra.dot(rb) <= std::cos(Pi / 180)) {
          parallax = true;
          break;
        }
      }
    if (!parallax)
      continue;
    selected.insert(id);
    for (const auto& o : point.track) {
      parent[root(o.image)] = root(point.track.front().image);
      const auto& im = model.images[o.image];
      const auto& camera = scene.cameras[im.camera];
      const auto& xy = im.pixels[o.feature];
      ++observations[o.image];
      cells[o.image].insert(camera.CoverageCell(xy));
      const double angle = std::acos(std::clamp(camera.Ray(xy).z(), -1., 1.));
      ++angles[im.camera][std::clamp(int(angle * 180 / Pi / 20), 0, 3)];
    }
  }
  std::vector<uint32_t> insufficient;
  bool connected = true;
  Json per_image = Json::array();
  for (uint32_t id = 0; id < model.images.size(); ++id)
    if (model.images[id].registered) {
      per_image.push_back({{"image_index", id},
                           {"observations", observations[id]},
                           {"occupied_cells", cells[id].size()}});
      if (observations[id] < MinimumImageObservations || cells[id].size() < MinimumImageCells)
        insufficient.push_back(id);
      connected = connected && root(id) == root(model.gauge_image);
    }
  report = {{"points", selected.size()},
            {"observations", std::accumulate(observations.begin(), observations.end(), size_t(0))},
            {"minimum_track_length", 3},
            {"minimum_angle_degrees", 1},
            {"maximum_mean_error_px", 2},
            {"minimum_image_observations", 32},
            {"minimum_image_cells", 8},
            {"minimum_supported_angle_bands", 3},
            {"minimum_observations_per_angle_band", 100},
            {"image_graph_connected", connected},
            {"observations_by_angle", angles},
            {"per_image", per_image},
            {"insufficient_images", insufficient}};
  if (!insufficient.empty()) {
    const auto id = insufficient.front();
    throw CalibrationCoverageIncomplete(
        insufficient, "Calibration lacks image coverage: " + scene.images[id].name + " (" +
                          std::to_string(observations[id]) + " observations, " +
                          std::to_string(cells[id].size()) + " cells)");
  }
  if (!connected)
    throw std::runtime_error("Calibration track graph is disconnected");
  for (uint32_t cid = 0; cid < angles.size(); ++cid) {
    const auto& bins = angles[cid];
    const bool used = std::any_of(model.images.begin(), model.images.end(), [&](const auto& image) {
      return image.registered && image.camera == cid;
    });
    if (used && std::count_if(bins.begin(), bins.end(), [](size_t n) { return n >= 100; }) < 3)
      throw std::runtime_error("Calibration needs at least three supported angle bands per lens");
  }
  return selected;
}
Reconstruction Reconstruct(const Scene& scene, const fs::path& work, int workers, Json& report,
                           bool stable_calibration, bool allow_partial, int ba_workers) {
  if (!ba_workers)
    ba_workers = workers;
  const auto started = Clock::now();
  Reconstruction m;
  const auto checkpoint = work / "cache/incremental.bin";
  double selection_seconds = 0, registration_seconds = 0, triangulation_seconds = 0,
         filter_seconds = 0, checkpoint_seconds = 0;
  auto filter = [&] {
    const auto phase = Clock::now();
    m.Filter();
    filter_seconds += Seconds(phase);
  };
  auto last_checkpoint = Clock::now();
  auto save_checkpoint = [&] {
    const auto phase = Clock::now();
    WriteModel(checkpoint, m);
    checkpoint_seconds += Seconds(phase);
    last_checkpoint = Clock::now();
  };
  const bool resumed = fs::exists(checkpoint);
  if (resumed) {
    m = ReadModel(checkpoint);
    if (m.images.size() != scene.images.size())
      throw std::runtime_error("Incremental image count differs");
    Log("Resume SfM at " + std::to_string(m.Registered()) + "/" + std::to_string(m.images.size()));
  } else {
    m.cameras = scene.cameras;
    m.images.resize(scene.images.size());
  }
  const auto colors = LoadFeatures(scene, work, m, resumed);
  const double cache_read_seconds = Seconds(started);
  const auto catalog = ReadJson(work / "cache/pairs.json");
  std::vector<std::array<uint32_t, 2>> candidates;
  for (const auto& pair : catalog.at("pairs"))
    candidates.push_back(pair.get<std::array<uint32_t, 2>>());
  const auto geometry_started = Clock::now();
  ImuPriorContext imu;
  imu.Initialize(scene);
  // Pair verification uses the whole CPU.
  const int geometry_workers =
      std::clamp(int(std::thread::hardware_concurrency()), workers, 32);
  const auto pairs = VerifyPairs(scene, m, work, candidates, geometry_workers, imu);
  std::vector<RelativeRotationPrior> rel_rot_priors = imu.BuildRelativeRotationPriors(scene);
  const double geometry_seconds = Seconds(geometry_started);
  const auto graph_started = Clock::now();
  Tracks graph(m, pairs);
  std::vector<uint64_t> points(graph.tracks.size(), NoPoint);
  std::vector<std::vector<size_t>> adjacency(m.images.size());
  for (size_t i = 0; i < pairs.size(); ++i)
    if (!pairs[i].inliers.empty()) {
      adjacency[pairs[i].a].push_back(i);
      adjacency[pairs[i].b].push_back(i);
    }
  Json optimization = Json::array();
  Json completion = Json::array();
  Json calibration_selection = Json::array();
  Json navigation_frame = {{"used", false}};
  const double graph_seconds = Seconds(graph_started);
  const auto positions = NavigationPositions(scene);
  // Empty until the model is moved into the navigation frame; then every BA holds the camera
  // centres to navigation and registration is gated by it.
  std::vector<Eigen::Vector3d> nav;
  // SfM bundle adjustment, with the navigation prior once it applies. Synchronous L/R poses are
  // not tied here: a global-shutter pose absorbs its own lens's readout distortion, so they are
  // not rigid (on 展厅 a rig-constrained SfM left the L readout time at zero later on, and a rig
  // pose hypothesis in registration lost three R exposures).
  auto ba = [&](const std::string& stage, int iterations, const std::set<uint32_t>& variable = {},
                const std::set<uint64_t>& selected = {}) {
    return Optimize(m, scene.cameras, stage, ba_workers, iterations, variable, {}, nav, selected,
                    nullptr, {}, {}, false, {}, rel_rot_priors);
  };
  auto unregister = [&](const std::vector<uint32_t>& ids) {
    for (uint32_t id : ids)
      m.images[id].registered = false;
    for (auto& point : m.points)
      point.track.erase(
          std::remove_if(point.track.begin(), point.track.end(),
                         [&](const auto& o) { return !m.images[o.image].registered; }),
          point.track.end());
    filter();
  };
  if (resumed) {
    for (size_t i = 0; i < graph.tracks.size(); ++i) {
      std::set<uint64_t> existing;
      for (const auto& o : graph.tracks[i])
        if (m.images[o.image].points[o.feature] != NoPoint)
          existing.insert(m.images[o.image].points[o.feature]);
      if (existing.size() == 1)
        points[i] = *existing.begin();
    }
  } else {
    const Pair* seed = nullptr;
    double best = 0;
    for (const auto& p : pairs)
      if (p.inliers.size() >= 100 && p.angle >= 1. &&
          (positions[p.a] - positions[p.b]).norm() > .2) {
        const double score = p.inliers.size() * std::min(p.angle, 8.);
        if (score > best) {
          seed = &p;
          best = score;
        }
      }
    if (!seed)
      throw std::runtime_error("No nondegenerate reconstruction seed");
    m.gauge_image = seed->a;
    m.scale_image = seed->b;
    m.gauge_baseline = 1;
    m.images[seed->a].registered = true;
    m.images[seed->b].registered = true;
    m.images[seed->b].pose = seed->relative;
    const auto triangulation_started = Clock::now();
    for (uint32_t i = 0; i < graph.tracks.size(); ++i)
      AddPoint(m, graph.tracks[i], points[i]);
    m.RebuildReferences();
    triangulation_seconds += Seconds(triangulation_started);
    optimization.push_back(ba("fixed", 100));
    filter();
    Log("Seed " + scene.images[seed->a].name + " + " + scene.images[seed->b].name);
  }
  size_t last_global = m.Registered();
  std::set<uint32_t> recent;
  if (!resumed)
    recent = {m.gauge_image, m.scale_image};
  std::array<size_t, 2> resumed_counts{};
  for (const auto& image : m.images)
    if (image.registered)
      ++resumed_counts[image.camera];
  bool calibrated =
      resumed && m.Registered() >= 32 && resumed_counts[0] >= 8 && resumed_counts[1] >= 8;
  // Move the visual model into the navigation frame once the registered exposures span at least
  // 0.3 m across their main direction, so the similarity is also fixed about that axis. Exposures
  // beyond the navigation gate are unregistered and may register again under the gate. A model
  // that is already in the navigation frame (a resumed checkpoint) fits to about the identity.
  auto to_navigation = [&] {
    std::vector<uint32_t> ids;
    std::vector<Eigen::Vector3d> centers, targets;
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (uint32_t id = 0; id < m.images.size(); ++id)
      if (m.images[id].registered) {
        ids.push_back(id);
        centers.push_back(m.images[id].pose.Center());
        targets.push_back(positions[id]);
        mean += positions[id];
      }
    if (ids.size() < 8)
      return;
    mean /= double(ids.size());
    Eigen::Matrix3d spread = Eigen::Matrix3d::Zero();
    for (const auto& p : targets)
      spread += (p - mean) * (p - mean).transpose();
    if (Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d>(spread / double(ids.size()))
            .eigenvalues()[1] < .3 * .3)
      return;
    Similarity fit;
    try {
      fit = RobustSimilarity(centers, targets, NavigationGate);
    } catch (const std::runtime_error&) {
      return;
    }
    std::vector<uint32_t> outliers;
    Json omitted = Json::array();
    for (size_t k = 0; k < ids.size(); ++k) {
      const double error = (fit.Apply(centers[k]) - targets[k]).norm();
      if (error > NavigationGate) {
        outliers.push_back(ids[k]);
        omitted.push_back({{"image_index", ids[k]}, {"error_m", error}});
      }
    }
    // A few outliers are wrong registrations; many mean the visual model has not settled yet.
    if (outliers.size() > ids.size() / 10 ||
        std::count(outliers.begin(), outliers.end(), m.gauge_image) ||
        std::count(outliers.begin(), outliers.end(), m.scale_image))
      return;
    m.Apply(fit);
    for (uint32_t id : outliers)
      Log("Unregister " + scene.images[id].name + " (beyond the navigation gate)");
    unregister(outliers);
    nav = positions;
    imu.RefreshFromNavigationModel(m);
    rel_rot_priors = imu.BuildRelativeRotationPriors(scene);
    navigation_frame = {{"used", true},
                        {"registered_images", ids.size()},
                        {"scale", fit.scale},
                        {"unregistered", omitted}};
    optimization.push_back(ba(calibrated ? "radial" : "fixed", 100));
    filter();
    Log("Navigation frame at " + std::to_string(m.Registered()) + " registered images");
  };
  to_navigation();
  while (m.Registered() < m.images.size()) {
    const auto selection_started = Clock::now();
    std::vector<std::pair<size_t, uint32_t>> options;
    for (uint32_t id = 0; id < m.images.size(); ++id)
      if (!m.images[id].registered)
        options.emplace_back(Correspondences(m, pairs, adjacency[id], id), id);
    std::sort(options.rbegin(), options.rend());
    selection_seconds += Seconds(selection_started);
    bool added = false;
    for (const auto& [count, id] : options) {
      if (count < 24)
        break;
      const auto registration_started = Clock::now();
      const bool success = Register(scene, m, pairs, adjacency[id], id,
                                    nav.empty() ? nullptr : &positions[id], imu, !nav.empty());
      registration_seconds += Seconds(registration_started);
      if (!success)
        continue;
      added = true;
      recent.insert(id);
      const auto triangulation_started = Clock::now();
      std::set<uint32_t> touched;
      for (uint32_t track : graph.lookup[id])
        if (track != UINT32_MAX)
          touched.insert(track);
      for (uint32_t track : touched)
        AddPoint(m, graph.tracks[track], points[track]);
      triangulation_seconds += Seconds(triangulation_started);
      if (nav.empty())
        to_navigation();
      const auto registered = m.Registered();
      if (registered % 8 == 0) {
        std::set<uint32_t> local = recent;
        std::map<uint32_t, size_t> connections;
        for (uint32_t image : recent)
          for (uint64_t point : m.images[image].points)
            if (point != NoPoint)
              for (const auto& o : m.points[point].track)
                if (!recent.count(o.image))
                  ++connections[o.image];
        std::vector<std::pair<size_t, uint32_t>> neighbors;
        for (const auto& [image, shared_points] : connections)
          neighbors.emplace_back(shared_points, image);
        std::sort(neighbors.rbegin(), neighbors.rend());
        for (size_t k = 0; k < std::min<size_t>(8, neighbors.size()); ++k)
          local.insert(neighbors[k].second);
        optimization.push_back(ba("fixed", 30, local));
        filter();
        recent.clear();
      }
      if (registered >= 32 && !calibrated) {
        std::array<size_t, 2> counts{};
        for (const auto& image : m.images)
          if (image.registered)
            ++counts[image.camera];
        if (counts[0] >= 8 && counts[1] >= 8) {
          optimization.push_back(ba("focal", 100));
          filter();
          completion.push_back(CompleteTracks(m, graph, points));
          optimization.push_back(ba("radial", 100));
          filter();
          completion.push_back(CompleteTracks(m, graph, points));
          calibrated = true;
          last_global = registered;
        }
      } else if (registered >= std::max(last_global + 40, size_t(last_global * 1.35))) {
        optimization.push_back(ba(calibrated ? "radial" : "fixed", 100));
        filter();
        completion.push_back(CompleteTracks(m, graph, points));
        last_global = registered;
      }
      if (registered % 16 == 0)
        Log("Registered " + std::to_string(registered) + "/" + std::to_string(m.images.size()));
      if (registered % 8 == 0 && Seconds(last_checkpoint) >= 20.0)
        save_checkpoint();
      break;
    }
    if (!added) {
      if (m.Registered() > last_global) {
        optimization.push_back(ba("radial", 150));
        filter();
        completion.push_back(CompleteTracks(m, graph, points));
        last_global = m.Registered();
        continue;
      }
      Json missing = Json::array();
      for (uint32_t id = 0; id < m.images.size(); ++id)
        if (!m.images[id].registered)
          missing.push_back(id);
      WriteJson(work / "cache/unregistered.json", missing);
      save_checkpoint();
      for (uint32_t id : missing.get<std::vector<uint32_t>>())
        Log("Unregistered " + scene.images[id].name + ": " +
            std::to_string(Correspondences(m, pairs, adjacency[id], id)) +
            " candidate observations");
      Log("SfM stalled at " + std::to_string(m.Registered()) + "/" +
          std::to_string(m.images.size()));
      if (!allow_partial)
        throw RegistrationIncomplete(missing.get<std::vector<uint32_t>>());
      Log("Finalize registered subset");
      break;
    }
  }
  // Growth has already refined focal length and distortion. Calibrate the
  // registered image set once, hold intrinsics while incorporating new tracks,
  // then check the calibration support again and refine cameras and points with
  // intrinsics held. A second full calibration there lowered the cost by under
  // 0.1% on 生产车间; the rolling-shutter stage refines the intrinsics again.
  Json calibration_completion = Json::array();
  auto calibrate = [&](const char* stage, int iterations) {
    for (;;) {
      filter();
      calibration_completion.push_back(CompleteUnassignedTracks(m, graph));
      Json support;
      std::set<uint64_t> selected;
      try {
        selected = CalibrationTracks(scene, m, support);
      } catch (const CalibrationCoverageIncomplete& missing) {
        support["used"] = false;
        calibration_selection.push_back(std::move(support));
        if (!allow_partial) {
          save_checkpoint();
          throw RegistrationIncomplete(missing.images);
        }
        // The usable subset must meet the same calibration limits. Remove
        // unsupported exposures and all their observations, then check again.
        for (uint32_t id : missing.images) {
          if (id == m.gauge_image || id == m.scale_image)
            throw std::runtime_error("Calibration cannot retain its reference images: " +
                                     scene.images[id].name);
          Log("Omit from final calibration: " + scene.images[id].name +
              " (insufficient reliable track coverage)");
        }
        unregister(missing.images);
        save_checkpoint();
        continue;
      }
      support["optimization_tracks"] = stable_calibration ? "stable" : "all";
      support["used"] = true;
      calibration_selection.push_back(std::move(support));
      if (!stable_calibration)
        selected.clear();
      optimization.push_back(ba(stage, iterations, {}, selected));
      break;
    }
  };
  calibrate("principal", 150);
  for (int round = 0; round < 3; ++round) {
    const size_t before = m.Observations();
    filter();
    const size_t removed = before - m.Observations();
    auto completed = CompleteTracks(m, graph, points);
    const size_t total_removed = removed + completed.at("removed_observations").get<size_t>();
    const size_t changed = total_removed + completed.at("added_observations").get<size_t>();
    completed["removed_observations"] = total_removed;
    completion.push_back(std::move(completed));
    if (round == 2 || (changed <= std::max<size_t>(1, size_t(before * .0005)) &&
                       optimization.back().at("converged").get<bool>()))
      break;
    optimization.push_back(ba("fixed", 100));
  }
  calibrate("fixed", 100);
  if (stable_calibration)
    optimization.push_back(ba("fixed", 200));
  filter();
  const auto unassigned_completion = CompleteUnassignedTracks(m, graph);
  const auto color_started = Clock::now();
  ColorPoints(m, colors);
  report = {{"verified_pairs", std::count_if(pairs.begin(), pairs.end(),
                                             [](const auto& p) { return !p.inliers.empty(); })},
            {"tracks", graph.tracks.size()},
            {"seconds", Seconds(started)},
            {"geometry_seconds", geometry_seconds},
            {"geometry_threads", geometry_workers},
            {"timing",
             {{"cache_read_seconds", cache_read_seconds},
              {"track_build_seconds", graph_seconds},
              {"candidate_selection_seconds", selection_seconds},
              {"pose_registration_seconds", registration_seconds},
              {"triangulation_seconds", triangulation_seconds},
              {"filter_seconds", filter_seconds},
              {"checkpoint_seconds", checkpoint_seconds},
              {"color_seconds", Seconds(color_started)}}},
            {"completion", completion},
            {"unassigned_completion", unassigned_completion},
            {"pre_calibration_completion", calibration_completion},
            {"navigation_frame", navigation_frame},
            {"imu_prior",
             {{"available", imu.available},
              {"used", imu.verified[0] || imu.verified[1]},
              {"verified", imu.verified},
              {"source", imu.source},
              {"boresight_median_deg", imu.boresight_median_deg},
              {"relative_rotation_priors", rel_rot_priors.size()},
              {"pairs_checked", imu.pairs_checked},
              {"pairs_rescued", imu.pairs_rescued},
              {"pairs_overruled", imu.pairs_overruled},
              {"pairs_rejected", imu.pairs_rejected},
              {"pnp_rescued", imu.pnp_rescued},
              {"pnp_overruled", imu.pnp_overruled},
              {"pnp_rejected", imu.pnp_rejected}}},
            {"calibration_tracks", stable_calibration ? "stable" : "all"},
            {"calibration_selection", calibration_selection},
            {"optimization", optimization}};
  if (m.Registered() == m.images.size())
    fs::remove(work / "cache/unregistered.json");
  else {
    Json missing = Json::array();
    for (uint32_t id = 0; id < m.images.size(); ++id)
      if (!m.images[id].registered)
        missing.push_back(id);
    WriteJson(work / "cache/unregistered.json", missing);
  }
  return m;
}
}  // namespace geosun
