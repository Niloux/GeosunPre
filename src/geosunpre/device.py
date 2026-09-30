"""Versioned Geosun L/R device recipe; scene selection lives in PipelineConfig."""

DEVICE_PROFILE = 'geosun_lr_v1'
FOV_DEGREES = 160
PRINCIPAL_Y_AXIS = 'up'
GUARD_REFERENCE_SIZE = (3000, 4000)
GUARD_BOTTOM_FRACTION = .8
GUARD_POLYGONS = {
    'L': [(1500, 0), (3000, 0), (3000, 1500), (2700, 1150), (2300, 1030),
          (2000, 1020), (1780, 860), (1500, 690)],
    'R': [(0, 0), (1700, 0), (1670, 700), (1470, 930), (1200, 1030),
          (900, 1100), (600, 1260), (350, 1340), (0, 1750)],
}
