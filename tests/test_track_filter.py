import numpy as np

from geosunpre.sfm.track_filter import triangle_components
from geosunpre.sfm.track_filter import training_adjacency


def test_closed_triangles_do_not_merge_through_single_ambiguous_bridge():
    nodes = [(f'image{i}', 0) for i in range(7)]
    graph = {n: {} for n in nodes}
    edges = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (2, 3), (5, 6)]
    for a, b in edges:
        graph[nodes[a]][nodes[b]] = 1.
        graph[nodes[b]][nodes[a]] = 1.
    components, triangles = triangle_components(nodes, graph)
    assert triangles == 2
    assert components == [nodes[:3], nodes[3:6]]
    assert triangle_components(nodes[:2], graph) == ([], 0)


def test_low_confidence_matches_cannot_supply_triangle_support(tmp_path):
    entries = []
    for a, b in [('a', 'b'), ('b', 'c'), ('a', 'c')]:
        path = tmp_path / f'{a}{b}.npz'
        np.savez(path, matches=np.array([[0, 0], [1, 1], [2, 2]]), scores=np.array([.95, .95, .2]))
        entries.append({'a': a, 'b': b, 'path': path})
    graph, _ = training_adjacency(entries, threshold=.5)
    assert ('c', 2) not in graph
    assert triangle_components([('a', 2), ('b', 2), ('c', 2)], graph) == ([], 0)
    assert triangle_components([('a', 1), ('b', 1), ('c', 1)], graph)[1] == 1
    assert ('a', 2) not in graph


def test_mapped_point_graph_keeps_only_edges_used_by_track_filter(tmp_path):
    entries = []
    for a, b in [('a', 'b'), ('b', 'c'), ('a', 'c')]:
        path = tmp_path / f'{a}{b}.npz'
        np.savez(path, matches=np.array([[0, 0], [1, 1]]), scores=np.array([.9, .9]))
        entries.append({'a': a, 'b': b, 'path': path})
    points = {('a', 0): 7, ('b', 0): 7, ('c', 0): 7,
              ('a', 1): 8, ('b', 1): 8, ('c', 1): 9}
    full, _ = training_adjacency(entries)
    limited, edges = training_adjacency(entries, mapped_points=points)
    assert edges == 4
    for nodes in [[('a', 0), ('b', 0), ('c', 0)], [('a', 1), ('b', 1)]]:
        assert triangle_components(nodes, limited) == triangle_components(nodes, full)
    assert ('c', 1) not in limited
