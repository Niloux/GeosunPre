"""Native fisheye geometry in COLMAP corner coordinates (first centre = 0.5)."""
from dataclasses import asdict, dataclass, replace

import cv2
import numpy as np


@dataclass(frozen=True)
class Camera:
    label: str
    width: int
    height: int
    params: tuple[float, ...]
    principal_offset_px: tuple[float, float]

    @property
    def K(self):
        fx, fy, cx, cy = self.params[:4]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    @property
    def D(self):
        return np.asarray(self.params[4:], dtype=np.float64)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(**{**value, 'params': tuple(value['params']),
                      'principal_offset_px': tuple(value['principal_offset_px'])})

    def with_principal_signs(self, x_sign=1, y_sign=-1):
        dx, dy = self.principal_offset_px
        params = (*self.params[:2], self.width / 2 + x_sign * dx,
                  self.height / 2 + y_sign * dy, *self.params[4:])
        return replace(self, params=params)

    def distorted_radius(self, fov_degrees):
        if not 0 < fov_degrees < 180:
            raise ValueError('This front-hemisphere pipeline requires 0 < FOV < 180 degrees')
        theta = np.deg2rad(fov_degrees / 2)
        samples = np.linspace(0, theta, 256)
        derivative = 1 + sum((2*i+3)*k*samples**(2*i+2) for i,k in enumerate(self.D))
        if np.any(derivative <= 0):
            raise ValueError(f'{self.label}: fisheye mapping is not monotonic in requested FOV')
        return theta * (1 + sum(k*theta**(2*i+2) for i,k in enumerate(self.D)))

    def valid(self, pixels, fov_degrees=160):
        p = np.asarray(pixels, dtype=np.float64)
        r2 = ((p[..., 0]-self.params[2])/self.params[0])**2
        r2 += ((p[..., 1]-self.params[3])/self.params[1])**2
        return (np.isfinite(p).all(axis=-1) & (r2 <= self.distorted_radius(fov_degrees)**2)
                & (p[..., 0] >= 0) & (p[..., 0] < self.width)
                & (p[..., 1] >= 0) & (p[..., 1] < self.height))

    def mask(self, fov_degrees=160):
        x = (np.arange(self.width, dtype=np.float64) + .5 - self.params[2])/self.params[0]
        y = (np.arange(self.height, dtype=np.float64) + .5 - self.params[3])/self.params[1]
        return (x[None, :]**2+y[:, None]**2 <= self.distorted_radius(fov_degrees)**2).astype(np.uint8)*255

    def rays(self, pixels):
        p = np.ascontiguousarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        if not len(p):
            return np.empty((0, 3), dtype=np.float64)
        xy = cv2.fisheye.undistortPoints(
            p, self.K, self.D, criteria=(cv2.TERM_CRITERIA_COUNT+cv2.TERM_CRITERIA_EPS, 50, 1e-12)
        ).reshape(-1, 2)
        b = np.column_stack((xy, np.ones(len(xy))))
        return b/np.linalg.norm(b, axis=1, keepdims=True)

    def project(self, xyz):
        xyz = np.ascontiguousarray(xyz, dtype=np.float64).reshape(-1, 1, 3)
        if not len(xyz):
            return np.empty((0, 2), dtype=np.float64)
        return cv2.fisheye.projectPoints(xyz, np.zeros(3), np.zeros(3), self.K, self.D)[0].reshape(-1, 2)

    def pycolmap(self):
        import pycolmap
        cam = pycolmap.Camera(model='OPENCV_FISHEYE', width=self.width,
                             height=self.height, params=list(self.params))
        cam.has_prior_focal_length = True
        return cam
