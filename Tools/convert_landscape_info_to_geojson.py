import json
import collections
import os

import numpy as np
from scipy.interpolate import interp1d, BPoly

GAP_CM = 100


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
        return hash(self.store['full_name'])


def cubic_example():
    x = np.linspace(0, 10, num=11, endpoint=True)
    y = np.cos(-x ** 2 / 9.0)
    f = interp1d(x, y)
    f2 = interp1d(x, y, kind='cubic')
    pass


FILENAME = 'landscape_segments.json'


# FILENAME = 'landscape_segments_mesa.json'

def main():
    with open('landscape_segments.json') as f:
        segments = json.load(f)
    segments = set(HashableSegment(s) for s in segments)
    visited_segments = set()
    segment_point_map = get_segment_point_map(segments)
    points = list(segment_point_map.keys())

    out = {
        'crosswalks': {},
        'driveways': {},
        'lanes': {},
        'lines': {},
        'origin': {'latitude': 0, 'altitude': 0, 'longitude': 0},
        'roadblocks': {},
        'stoplines': {},
    }

    # landscape_offset = np.array([-11200.000000, -11200.000000, 100.000000])
    landscape_offset = np.array([-21214.687500, -21041.564453, 18179.121094])  # TODO: Get this from Unreal
    landscape_z_scale = np.array([1, 1, 159.939941 / 100])

    all_center_points = []

    for point in points:
        actual = landscape_to_world(landscape_offset, landscape_z_scale, point)
        print('actual', format_point_as_unreal(actual))
        print('point', format_point_as_unreal(point))
        for segment in segment_point_map[point]:
            if segment in visited_segments:
                continue
            else:
                visited_segments.add(segment)
            # rights = interpolate(landscape_offset, segment, 'Right')
            center_points = get_interpolated_points(landscape_offset, landscape_z_scale,
                                                    point, segment, 'Center')
            segment['interp_center_points'] = center_points

            # TODO: Use the guard rails to get the right and left edges of the road

            # TODO: Eventually we need to start with the map and build the level
            #       OR if we want to extract the map from landscape splines, we can use the mesh name
            #       to know the lanes and directions, then do a raycast search for the edges of the road
            #       starting from the center (either straight down or along the normal to the plane formed
            #       by the segment), so we are not reliant on matching Unreal's interpolation and
            #       we don't have to correct for Unreal's width segment connection tangents and control point tangents.

            # interpolate(distances, landscape_offset, segment, 'Center')
            # interpolate(distances, landscape_offset, segment, 'Left')

    # DONE: For each point in the segment_point map
    # DONE: Check that interp points make sense in the editor
    # DONE: Use left and right to get left and right road edge from interp points
    # DONE: Spot check above
    # DONE: Do cubic interpolation on those points to get 1m increments
    # DONE: Visualize results
    # TODO: String segments
    # TODO: Assign drive paths to points in between center and right, left edges of road
    # TODO: Add the previous and next nodes per JSON map spec
    # TODO: Add adjacent lanes (not yet in JSON spec but will be)
    # TODO: Subtract the Landscape center from the points
    # TODO: Use interpolated points to and cubic spline interpolation to get points sep by 1m
    # graphs =  get_map_graphs(graphs, points, segment_point_map)
    # print('Num splines', len(graphs))


USE_TANGENTS = False


def get_interpolated_points(landscape_offset, landscape_z_scale, point, segment, side_name):
    if not USE_TANGENTS:
        return interpolate(landscape_offset, landscape_z_scale, segment, side_name)
    else:
        side_points, tangents = get_side_points_and_tangents(landscape_offset, landscape_z_scale,
            segment, side_name)

        if tangents is None:
            return side_points
        else:
            new_points = sample_cubic_splines_with_derivative(side_points, tangents, GAP_CM)

            if np.isclose(point[0], 9730.796875) or np.isclose(point[0], 10259.5302734375):
                for point in new_points:
                    print('check it!', format_point_as_unreal(point))
            return new_points


def interpolate(landscape_offset, landscape_z_scale, segment, side):
    """

    :param landscape_offset: Center of landscape which Unreal uses as origin for segment point coordinates
    :param landscape_offset: Z-scale of landscape in Unreal transform
    :param segment: Unreal segment
    :param side: Left, Center, or Right
    :return:
    """

    # TODO: Vectorize all operations in this method

    orig = get_side_points(landscape_offset, landscape_z_scale, segment, side)

    if len(orig) <= 2:
        return np.array(orig)
    else:
        # Do cubic interpolation to smooth out inner points

        if len(orig) < 4:
            points = add_midpoints(orig)
        else:
            points = orig

        distances = np.cumsum(
            np.linalg.norm(np.diff(points, axis=0), axis=1))
        distances = np.hstack([[0], distances])

        points = np.array(points)
        total_points = distances[-1] // GAP_CM + 1
        chords = list(np.arange(0, total_points) * GAP_CM)
        if distances[-1] % GAP_CM > (GAP_CM / 2):
            # Add an extra interp point so there won't be a gap longer than meter
            chords.append(distances[-1])
        cubic_interpolator = interp1d(distances, points, 'cubic', axis=0)
        ret = cubic_interpolator(chords)
        for i in range(len(ret) - 1):
            pt = ret[i]
            next_pt = ret[i + 1]
            print(format_point_as_unreal(pt))
            print('distance', np.linalg.norm(next_pt - pt))
        print('')
        return ret


def sample_cubic_splines_with_derivative(points, tangents, resolution):
    """
    Compute and sample the cubic splines for a set of input points with
    optional information about the tangent (direction AND magnitude). The
    splines are parametrized along the traverse line (piecewise linear), with
    the resolution being the step size of the parametrization parameter.
    The resulting samples have NOT an equidistant spacing.

    Arguments:      points: a list of n-dimensional points
                    tangents: a list of tangents
                    resolution: parametrization step size
    Returns:        samples

    Notes: Lists points and tangents must have equal length. In case a tangent
           is not specified for a point, just pass None. For example:
                    points = [[0,0], [1,1], [2,0]]
                    tangents = [[1,1], None, [1,-1]]

    """
    resolution = float(resolution)
    points = np.asarray(points)
    n_points, dim = points.shape

    # Parametrization parameter s.
    dp = np.diff(points, axis=0)  # difference between points
    dp = np.linalg.norm(dp, axis=1)  # distance between points
    d = np.cumsum(dp)  # cumsum along the segments
    d = np.hstack([[0], d])  # add distance from first point
    length = d[-1]  # length of point sequence
    n_samples = int(length / resolution)  # number of samples

    # TODO: s should be GAP_CM apart plus left over distance to end if over GAP_CM / 2

    s, r = np.linspace(0, length, n_samples, retstep=True)  # sample parameter and step

    # Bring points and (optional) tangent information into correct format.
    assert (len(points) == len(tangents))
    data = np.empty([n_points, dim], dtype=object)
    for i, p in enumerate(points):
        t = tangents[i]
        # Either tangent is None or has the same
        # number of dimensions as the point p.
        assert (t is None or len(t) == dim)
        fuse = list(zip(p, t) if t is not None else zip(p, ))
        data[i, :] = fuse

    # Compute splines per dimension separately.
    samples = np.zeros([n_samples, dim])
    for i in range(dim):
        poly = BPoly.from_derivatives(d, data[:, i])
        samples[:, i] = poly(s)
    return samples


def get_side_points(landscape_offset, landscape_z_scale, segment, side):
    orig = []
    for p in segment['Points']:
        p = landscape_to_world(landscape_offset, landscape_z_scale, p[side])
        orig.append(p)
    return orig


def get_segment_tangent_at_point(segment, point):
    point = tuple(point)
    for i, p in enumerate(segment['SplineInfo_Points']):
        if get_segment_point(segment, i) == point:
            arrive = p['ArriveTangent']
            leave = p['LeaveTangent']
            if arrive != leave:
                raise NotImplementedError('Unequal tangents not supported')
            return np.array(arrive)

def get_side_points_and_tangents(landscape_offset, segment, side_name):
    points = []
    # TODO: If there are three points, average the tangent of the two end points for the center tangent
    # TODO: Also generate a point between each endpoint and the center whose tangent matches the endpoint
    orig = segment['Points']
    if len(orig) == 2:
        return orig, None
    elif len(orig) == 3:
        begin_tangent = get_segment_tangent_at_point(segment, orig[0]['Center'])
        end_tangent = get_segment_tangent_at_point(segment, orig[2]['Center'])
        mid_tangent = (begin_tangent + end_tangent) / 2
        for i, p in enumerate(orig):
            p = np.array(p[side_name]) + landscape_offset
            points.append(p)
        points = add_midpoints(points)
        tangents = np.array([begin_tangent, begin_tangent, mid_tangent, end_tangent, end_tangent])
        tangents *= 0
    else:
        raise ValueError('Expected either 2 or 3 points in segment, got %d' % len(orig))
    return np.array(points), np.array(tangents)


def add_midpoints(orig):
    ret = []
    for i in range(len(orig) - 1):
        start = orig[i]
        end = orig[i + 1]
        mid = start + (end - start) * 0.5
        ret += [start, mid]
    ret.append(orig[-1])
    return ret


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
                segment_point_map[point].add(s)
            else:
                segment_point_map[point] = {s}
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
