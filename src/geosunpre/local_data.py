"""Shared whole-track holdout policy for local and global SfM."""
import numpy as np
from .frontend import FrontendConfig
from .geometry import build_tracks
from .scene import digest

CONFIG = FrontendConfig(detector_size=1536, num_keypoints=3072)

def local_pairs(images):
    pairs = []
    for i,a in enumerate(images):
        for b in images[i+1:]:
            gap = abs(a['frame_index']-b['frame_index'])
            same = a['camera']==b['camera']
            if (same and (1<=gap<=5 or (gap in (8,12) and a['frame_index']%3==0))) or (not same and gap<=2):
                pairs.append((a,b))
    return pairs

def split_tracks(pair_data, images, fraction=.2, seed=314159):
    """Reserve complete raw LoMa tracks before any camera geometry is fitted.

    Every edge touching any reserved observation is excluded from training,
    including edges rejected by the conflict-aware raw-track union.
    """
    summary, tracks = build_tracks(pair_data,images)
    names = {im['name']:im for im in images}
    selected, held = [], set()
    for track in tracks:
        if len(track)<4:
            continue
        frames = [names[name]['frame_index'] for name,index in track]
        if max(frames)-min(frames)<2:
            continue
        token = digest({'seed':seed,'track':track})
        if int(token[:16],16)/2**64 >= fraction:
            continue
        # Supports depend only on time/identity, never reprojection error. All
        # remaining observations are predicted after triangulating these two.
        ordered = sorted(track,key=lambda n:(names[n[0]]['frame_index'],n[0],n[1]))
        support = [ordered[0],ordered[-1]]
        queries = [n for n in ordered if n not in support]
        selected.append({'id':token[:16],'observations':track,'support':support,'queries':queries})
        held.update((name,int(idx)) for name,idx in track)
    return {'raw_track_summary':summary,'fraction':fraction,'seed':seed,
            'tracks':selected,'heldout_observations':len(held),
            'split_before_geometry':True,'scope':'Matched-track holdout, not independent acquisition or ground truth.'}, held

def train_edge_mask(a,b,matches,held):
    return np.array([(a,int(i)) not in held and (b,int(j)) not in held for i,j in matches],dtype=bool)
