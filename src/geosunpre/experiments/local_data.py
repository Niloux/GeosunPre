"""Compatibility imports for historical local and transfer experiments."""

from ..local_data import CONFIG, local_pairs, split_tracks, train_edge_mask
from ..pipeline.local_calibration import (LOCAL_VERSION, SEGMENTS, build_frontend,
                                          load_local, prepare_masks)
