#pragma once

namespace geosun {
// Smallest-eigenvalue eigenvector of a symmetric 3x3 matrix by cyclic Jacobi rotations;
// values returns the eigenvalues in ascending order.
inline __device__ void SymmetricEigen(double a[3][3], double values[3], double3& normal) {
  double v[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  for (int sweep = 0; sweep < 12; ++sweep) {
    const double off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    if (off < 1e-30)
      break;
    for (int p = 0; p < 2; ++p)
      for (int q = p + 1; q < 3; ++q) {
        if (fabs(a[p][q]) < 1e-300)
          continue;
        const double theta = (a[q][q] - a[p][p]) / (2 * a[p][q]),
                     t = (theta >= 0 ? 1 : -1) / (fabs(theta) + sqrt(theta * theta + 1)),
                     cs = 1 / sqrt(t * t + 1), sn = t * cs;
        for (int k = 0; k < 3; ++k) {
          const double kp = a[k][p], kq = a[k][q];
          a[k][p] = cs * kp - sn * kq;
          a[k][q] = sn * kp + cs * kq;
        }
        for (int k = 0; k < 3; ++k) {
          const double pk = a[p][k], qk = a[q][k];
          a[p][k] = cs * pk - sn * qk;
          a[q][k] = sn * pk + cs * qk;
        }
        for (int k = 0; k < 3; ++k) {
          const double kp = v[k][p], kq = v[k][q];
          v[k][p] = cs * kp - sn * kq;
          v[k][q] = sn * kp + cs * kq;
        }
      }
  }
  int index[3] = {0, 1, 2};
  for (int i = 0; i < 3; ++i)
    for (int j = i + 1; j < 3; ++j)
      if (a[index[j]][index[j]] < a[index[i]][index[i]]) {
        const int swap = index[i];
        index[i] = index[j];
        index[j] = swap;
      }
  for (int i = 0; i < 3; ++i)
    values[i] = a[index[i]][index[i]];
  normal = make_double3(v[0][index[0]], v[1][index[0]], v[2][index[0]]);
}
}  // namespace geosun
