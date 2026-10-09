"""Bounded-memory pair retrieval and repair using measured match support."""

import numpy as np


class PairSelection:
    """Own the candidate graph; callers only supply a matching callback.

    Appearance and raw match support are proposals, not geometric validation.
    Final camera-aware verification remains in mapping after calibration.
    """

    def __init__(self, images, names, descriptors):
        self.images = images
        self.names = [r['name'] for r in images]
        if set(map(str, names)) != set(self.names):
            raise ValueError('Retrieval descriptors do not cover the selected images')
        lookup = {str(name): i for i, name in enumerate(names)}
        self.descriptors = np.asarray(descriptors, dtype=np.float32)[[lookup[n] for n in self.names]]
        self.times = np.array([r['week'] * 604800 + r['seconds'] for r in images])
        self.sequences = {}
        for camera in sorted({r['camera'] for r in images}):
            ids = [i for i, r in enumerate(images) if r['camera'] == camera]
            self.sequences[camera] = np.array(sorted(ids, key=lambda i: (self.times[i], self.names[i])))
        self.pairs = {}

    def _add(self, a, b, reason):
        key = tuple(sorted((int(a), int(b))))
        if key[0] != key[1]:
            self.pairs.setdefault(key, set()).add(reason)

    def _blocks(self):
        for start in range(0, len(self.images), 128):
            yield start, self.descriptors[start:start + 128] @ self.descriptors.T

    @staticmethod
    def _top(scores, ids, count):
        return ids[np.lexsort((ids, -scores[ids]))[:count]]

    def _initial(self):
        for ids in self.sequences.values():
            for position, a in enumerate(ids):
                for b in ids[position + 1:position + 6]:
                    self._add(a, b, 'temporal')
        for a, record in enumerate(self.images):
            for camera, ids in self.sequences.items():
                if camera == record['camera']:
                    continue
                centre = int(np.searchsorted(self.times[ids], self.times[a]))
                for b in ids[max(0, centre - 2):centre + 3]:
                    self._add(a, b, 'nearby_exposure')
        for start, block in self._blocks():
            for offset, scores in enumerate(block):
                a = start + offset
                for ids in self.sequences.values():
                    centre = int(np.searchsorted(self.times[ids], self.times[a]))
                    gap = np.abs(np.arange(len(ids)) - centre)
                    eligible = ids[gap > 5]
                    for b in self._top(scores, eligible, 2):
                        self._add(a, b, 'appearance_retrieval')
                    # One candidate at each chronological scale preserves long
                    # baselines without assuming that a route closes at its ends.
                    bins = np.floor(np.log2(np.maximum(gap, 1))).astype(int)
                    for bucket in np.unique(bins[gap > 5]):
                        candidates = ids[(gap > 5) & (bins == bucket)]
                        for b in self._top(scores, candidates, 1):
                            self._add(a, b, 'multiscale_retrieval')

    def _graph(self, supported):
        parent = np.arange(len(self.images))
        degree = np.zeros(len(parent), dtype=int)

        def root(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return int(a)

        for a, b in sorted(supported):
            ra, rb = root(a), root(b)
            parent[max(ra, rb)] = min(ra, rb)
            degree[a] += 1
            degree[b] += 1
        labels = np.array([root(a) for a in range(len(parent))])
        _, counts = np.unique(labels, return_counts=True)
        summary = {'components': len(counts), 'component_sizes': sorted(counts.tolist(), reverse=True),
                   'images_below_three_supported_pairs': int((degree < 3).sum()),
                   'supported_pairs': len(supported)}
        return labels, degree, summary

    def _repair(self, labels, degree, tried):
        bridges = {}
        repairs = set()
        all_ids = np.arange(len(self.images))
        for start, block in self._blocks():
            for offset, scores in enumerate(block):
                a = start + offset
                available = np.ones(len(all_ids), dtype=bool)
                available[a] = False
                for b in tried[a]:
                    available[b] = False
                if degree[a] < 3:
                    for b in self._top(scores, all_ids[available], 4):
                        repairs.add(tuple(sorted((a, int(b)))))
                external = all_ids[available & (labels != labels[a])]
                for b in self._top(scores, external, 2):
                    key = tuple(sorted((a, int(b))))
                    choices = bridges.setdefault(int(labels[a]), {})
                    choices[key] = float(scores[b])
        for choices in bridges.values():
            repairs.update(sorted(choices, key=lambda key: (-choices[key], key))[:2])
        return sorted(repairs)

    def match(self, matcher):
        """Match initial proposals, then at most two bounded graph repair rounds.

        matcher(pair) returns (cache entry, count of raw matches above 0.5).
        Frozen results contain decisions and support, independent of cache hits.
        """
        self._initial()
        initial_count = len(self.pairs)
        entries, supported = {}, set()
        tried = [set() for _ in self.images]

        def run(keys):
            for a, b in keys:
                pair = {'a': self.names[a], 'b': self.names[b], 'reasons': sorted(self.pairs[a, b])}
                entry, support = matcher(pair)
                entries[a, b] = entry
                tried[a].add(b)
                tried[b].add(a)
                if support >= 30:
                    supported.add((a, b))

        run(sorted(self.pairs))
        labels, degree, before = self._graph(supported)
        after = before
        rounds = []
        for _ in range(2):
            if after['components'] == 1 and not np.any(degree < 3):
                break
            additions = self._repair(labels, degree, tried)
            if not additions:
                break
            for a, b in additions:
                self._add(a, b, 'match_graph_repair')
            run(additions)
            labels, degree, after = self._graph(supported)
            rounds.append({'additional_pairs': len(additions), **after})
        report = {'version': 2, 'initial_pairs': initial_count,
            'final_pairs': len(entries), 'graph_before_repair': before, 'graph_after_repair': after,
            'repair_rounds': rounds, 'similarity_block_rows': 128,
            'support': 'At least 30 raw matches with confidence >= 0.5; not geometric verification.',
            'selection_uses_input_pose_or_las': False}
        return [entries[key] for key in sorted(entries)], report
