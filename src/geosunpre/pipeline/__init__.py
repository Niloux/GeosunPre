"""The supported input-to-COLMAP workflow."""

VALIDATION_STAGES = ('prepare', 'masks', 'features', 'matches', 'holdout',
                     'calibration', 'sfm', 'finalize', 'export')
PRODUCTION_STAGES = tuple('observations' if s == 'holdout' else s for s in VALIDATION_STAGES)
STAGES = (*PRODUCTION_STAGES, 'holdout')
