import json
import numpy as np

import collections


class HashableSegment(collections.MutableMapping):
    def __init__(self, segment):
        self.store = segment
        self.update(segment)

    def __getitem__(self, key):
        return self.store[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def __hash__(self):
        return self.store['full_name']


def main():
    with open('landscape_segments_mesa_hill.json') as f:
        segments = json.load(f)
    segments = get_unique_segments(segments)  # Segments seem already de-duped, but just in case
    segments = [HashableSegment(s) for s in segments]
    segment_point_map = get_segment_point_map(segments)
    points = list(segment_point_map.keys())

    landscape_offset = np.array([-11200.000000, -11200.000000, 100.000000])  # TODO: Add this from JSON
    for point in points:
        actual = np.array(point) + landscape_offset
        print('actual', format_point_as_unreal(actual))
        print('point', format_point_as_unreal(point))
        for segment in segment_point_map[point]:
            for interp_point in segment['Points']:
                print('interp right', format_point_as_unreal(
                    np.array(interp_point['Right']) + landscape_offset))

    # DONE: For each point in the segment_point map
    # DONE: Check that interp points make sense in the editor
    # DONE: Use left and right to get left and right road edge from interp points
    # DONE: Spot check above
    # TODO: Do cubic interpolation on those points to get 1m increments
    # TODO: Visualize results
    # TODO: Assign drive paths to points in between center and right, left edges of road
    # TODO: Add the previous and next nodes per JSON map spec
    # TODO: Add adjacent lanes (not yet in JSON spec but will be)
    # TODO: Subtract the Landscape center from the points
    # TODO: Use interpolated points to and cubic spline interpolation to get points sep by 1m
    #graphs =  get_map_graphs(graphs, points, segment_point_map)
    # print('Num splines', len(graphs))


def format_point_as_unreal(point):
    return '(X=%.6f,Y=%.6f,Z=%.6f)' % (point[0], point[1], point[2])


def get_map_graphs(points, segment_point_map):
    graphs = []
    while points:
        # Find overlapping segments using OutVal of control point
        # String mid points together    unique_segments = dict()
        start_point = list(segment_point_map.items())[0]
        del segment_point_map[start_point]
        graph = get_map_graph(segment_point_map, start_point, points)
        graphs.append(graph)
    return graphs


def get_segment_point_map(segments):
    segment_point_map = dict()
    for s in segments:
        for point in get_segment_points(s):
            if point in segment_point_map:
                segment_point_map[point].append(s)
            else:
                segment_point_map[point] = [s]
    return segment_point_map


def get_segment_points(s):
    return [get_segment_point(s, i) for i in range(len(s['SplineInfo_Points']))]


def get_unique_segments(segments):
    unique_segments = dict()
    for segment in segments:
        unique_segments[segment['full_name']] = segment
    segments = list(unique_segments.values())
    return segments


def get_neighboring_points(p, segment_point_map):
    segments = segment_point_map[p]
    for s in segments:
        for pt in get_segment_points(s):
            if pt != p:
                yield pt
        else:
            raise RuntimeError('Could not match segment')
    else:
        print('Only one segment with this control point')
        yield None


def get_map_graph(segment_point_map, start_point, points):
    # TODO BFS here


    # def breadth_first_search(graph, root):
    #     visited, queue = set(), collections.deque([root])
    #     while queue:
    #         vertex = queue.popleft()
    #         for neighbour in graph[vertex]:
    #             if neighbour not in visited:
    #                 visited.add(neighbour)
    #                 queue.append(neighbour)

    visited = set()
    q = collections.deque([start_point])
    graph = []

    # Use BFS to find connected points
    while q:
        p = q.popleft()
        for p in get_neighboring_points(p, segment_point_map):
            if p not in visited:
                visited.add(p)
                q.append(p)
                points.remove(p)  # Don't start a search from here

                # graph

        # for p in get_segment_points(s):

    return graph


def get_segment_point(segment, index):
    return tuple(segment['SplineInfo_Points'][index]['OutVal'])


if __name__ == '__main__':
    main()
