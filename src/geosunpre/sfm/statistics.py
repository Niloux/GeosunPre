"""Paired summaries for frozen SfM validation queries."""

import numpy as np


def paired_track_bootstrap(values, common, baseline='fixed_calibration', replicates=1000):
    """Uncertainty of paired error changes, resampling complete heldout tracks."""
    base = values[baseline]
    result = {}
    for name, value in values.items():
        selected = common & np.isfinite(value['error']) & np.isfinite(base['error'])
        track_ids = np.unique(base['track_id'][selected])
        deltas = value['error'][selected] - base['error'][selected]
        selected_tracks = base['track_id'][selected]
        chunks = [deltas[selected_tracks == track] for track in track_ids]
        if not chunks:
            result[name] = {'queries': 0, 'tracks': 0, 'median_delta_px': None,
                            'interval95_px': None}
            continue
        rng = np.random.default_rng(271829)
        medians = [float(np.median(np.concatenate(
            [chunks[index] for index in rng.integers(len(chunks), size=len(chunks))])))
            for _ in range(replicates)]
        result[name] = {
            'queries': int(selected.sum()), 'tracks': len(chunks),
            'median_delta_px': float(np.median(np.concatenate(chunks))),
            'interval95_px': np.quantile(medians, [.025, .975]).tolist(),
            'replicates': replicates, 'seed': 271829,
            'note': 'Conditional paired median error change on finite common-support queries; not calibration uncertainty or a ground-truth accuracy interval.',
        }
    return result
