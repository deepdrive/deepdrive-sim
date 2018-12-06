import copy
import json
import collections
import os

import numpy as np
from scipy import spatial
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
    # DONE: For each point in the segment_point map
    # DONE: Check that interp points make sense in the editor
    # DONE: Use left and right to get left and right road edge from interp points
    # DONE: Spot check above
    # DONE: Do cubic interpolation on those points to get 1m increments
    # DONE: Visualize results
    # DONE: String segments
    # TODO: Assign drive paths to points in between center and right, left edges of road
    # TODO: Add the previous and next nodes per JSON map spec
    # TODO: Add adjacent lanes (not yet in JSON spec but will be)
    # DONE: Subtract the Landscape center from the points
    # DONE: Use interpolated points to and cubic spline interpolation to get points sep by 1m
    with open('landscape_segments.json') as f:
        segments = json.load(f)
    segments = set(HashableSegment(s) for s in segments)
    segment_point_map = get_segment_point_map(segments)
    points = list(segment_point_map.keys())
    graphs = get_map_graphs(points, segment_point_map)
    graphs_by_side = match_graphs_to_side(graphs)
    side_point_mapping = match_points_to_sides(graphs_by_side, points)
    sides = interpolate_points_by_segment(points, segment_point_map, side_point_mapping)
    get_lanes(sides, graphs_by_side)
    with open('deepdrive-canyons-map.json', 'w') as outfile:
        json.dump(serialize_as_geojson(segments), outfile)
        print('Saved map to %s' % os.path.realpath(outfile.name))


def get_closest_point(point, kd_tree):
    distance, index = kd_tree.query(point)
    point = kd_tree.data[index]
    return point, distance


def get_lanes(sides, graphs_by_side):
    # Start in a known direction along center
    # Find closest right / left to center not visited, then mark as visited
    # If visited, raise exception and check it out
    # Else get midpoints and use those for drive path points
    # Reverse direction for opposite lane
    # for p
    inner_lanes = []
    outer_lanes = []
    start_point = (-27059.478515625, 22658.291015750001, 20553.831906106912)  #  (-5844.791015625, 43699.85546875, 1484.7515869140625)
    second_point = (-26989.210175210432, 22727.784224634106, 20537.975077496125) #  (-1131.97998046875, 52382.1171875, 613.6241455078125)
    end_point = (-27128.758053206424, 22593.083229395124, 20568.862235249715)  # (-8673.3046875, 42718.24609375, 1666.61572265625)
    center = start_point
    prev = end_point
    prev_inner = None
    prev_outer = None
    visited_inner = set()
    visited_outer = set()
    visited_center = set(start_point)
    inner_kd_tree = spatial.KDTree(sides['inner'])
    outer_kd_tree = spatial.KDTree(sides['outer'])
    while True:
        inner, inner_dist = get_closest_point(center, inner_kd_tree)
        outer, outer_dist = get_closest_point(center, outer_kd_tree)

        if max(inner_dist, outer_dist) > 15 * 100:
            raise ValueError('Got lane width way wider than expected')

        if prev_inner is None or prev_outer is None:
            print('Boostrapping previous points')
        else:
            inner_lane = get_lane(center, inner, prev_inner, visited_inner, is_opposite=False)
            outer_lane = get_lane(center, outer, prev_outer, visited_outer, is_opposite=True)
            # TODO: Add adjacent lane references
            inner_lanes.append(inner_lane)
            outer_lanes.append(outer_lane)

        prev_inner = inner
        prev_outer = outer

        for neighbor in graphs_by_side['center'][center]:
            if neighbor != prev:
                center = neighbor

        if center == start_point:
            break


def get_lane(left, right, prev, visited, is_opposite):
    if right in visited:
        raise ValueError('Right side of lane point is the closest point to two center points')
    drive = left + (right - left) * 0.5
    if is_opposite:
        vector = prev - right
    else:
        vector = right - prev
    direction = np.arctan(vector[1] / vector[0])
    lane = {'left': left, 'drive': drive, 'right': right, 'direction': direction}
    visited.add(right)
    return lane


"""
-21348.18270874  30332.50195325  19115.9702746
"""

def interpolate_points_by_segment(points, segment_point_map, side_point_mapping):
    visited_segments = set()
    # landscape_offset = np.array([-11200.000000, -11200.000000, 100.000000])  # Mesa offset
    landscape_offset = np.array([-21214.687500, -21041.564453, 18179.121094])  # Canyons TODO: Get this from Unreal
    landscape_z_scale = np.array([1, 1, 159.939941 / 100])
    sides = collections.defaultdict(list)
    for point in points:
        if point not in side_point_mapping:
            print('Ignoring point %r - not on track and should be deleted')
            continue
        for segment in segment_point_map[point]:
            if segment in visited_segments:
                continue
            else:
                visited_segments.add(segment)
            center_points = get_interpolated_points(landscape_offset, landscape_z_scale, point, segment, 'Center')
            segment['interp_center_points'] = center_points
            sides[side_point_mapping[point]].extend(center_points)  # Guard rails define inner and outer road edges
            # TODO: Eventually we need to start with the map and build the level
            #       OR if we want to extract the map from landscape splines, we can use the mesh name
            #       to know the lanes and directions, then do a raycast search for the edges of the road
            #       starting from the center (either straight down or along the normal to the plane formed
            #       by the segment), so we are not reliant on matching Unreal's interpolation and
            #       we don't have to correct for Unreal's width segment connection tangents and control point tangents.

    return sides


def match_points_to_sides(graphs_by_side, points):
    side_point_mapping = {}
    for p1 in points:
        for s in graphs_by_side:
            if p1 in graphs_by_side[s]:
                side_point_mapping[p1] = s
    return side_point_mapping


def match_graphs_to_side(graphs):
    sides = {}
    for g in graphs:
        if (16453.779296875, 16242.0615234375, 1334.0733642578125) in g:
            sides['inner'] = g
        elif (25330.23046875, 56882.2890625, 2772.235595703125) in g:
            sides['center'] = g
        elif (25125.541015625, 55631.3828125, 2850.599365234375) in g:
            sides['outer'] = g

    return sides


def get_landscape_spline_name(segment):
    seg_name = segment['full_name']
    comps = [c for c in segment['full_name'].split('.') if 'component' in c.lower()]
    if len(comps) != 1:
        raise ValueError('Could not get landscape spline name from %s' % seg_name)
    return comps[0]

def landscape_to_world(landscape_offset, landscape_z_scale, point):
    point_with_z_scale = point * landscape_z_scale
    actual = np.array(point_with_z_scale) + landscape_offset
    return actual

def serialize_as_geojson(in_segments):
    out = {
        'crosswalks': {},
        'driveways': {},
        'lanes': {},
        'lines': {},
        'origin': {'latitude': 0, 'altitude': 0, 'longitude': 0},
        'roadblocks': {},
        'stoplines': {},
    }
    lane_segments = []
    out_segment_id = 0
    for s in in_segments:
        points = s['interp_center_points']
        for i in range(len(points) - 1):
            start = points[i]
            end = points[i + 1]
            vector = end - start
            lane_segments.append(
                get_geojson_lane_segment(end, out_segment_id, start, vector))
            out_segment_id += 1
    out['lanes'] = {'features': lane_segments, 'type': 'FeatureCollection'}
    return out


def get_geojson_lane_segment(end, segment_id, start, vector):
    return {
        'geometry': {'type': 'LineString', 'coordinates': [start.tolist(), end.tolist()]},
        'properties': {
            'distance': np.linalg.norm(vector),
            'direction': np.arctan(vector[1] / vector[0]),
            'vehicle_type': "VEHICLE",
            'id': segment_id,
            'intersections': [],
            'lane_count': 1,
            'lane_num': 1,
            'lane_type': "STRAIGHT",
            'left_line': {},
            'next_lanes': [],
            'other_attributes': [],
            # 'polygon': {type: "Polygon", coordinates: [[-87.6406294199376, 115.43642618666914, 125.653404236],…]},
            'prev_lanes': [],
            'radius': 0,
            # 'right_line': {, …},
            'speed_limit': 60,
        },
        'type': "Feature"}


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

def get_side_points_and_tangents(landscape_offset, landscape_z_scale, segment, side_name):
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
            p = landscape_to_world(landscape_offset, landscape_z_scale, np.array(p))
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
    points = copy.deepcopy(points)
    while points:
        # Find overlapping segments using OutVal of control point
        # String mid points together    unique_segments = dict()
        graph = get_map_graph(segment_point_map, points)
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
    ret = []
    for s in segments:
        for pt in get_segment_points(s):
            if pt != p:
                ret.append(pt)
    return ret


def get_map_graph(segment_point_map, points):
    # Use BFS to find connected points
    visited = {points[0]}
    q = collections.deque([points[0]])
    graph = {}
    while q:
        vertex = q.popleft()
        points.remove(vertex)  # Don't start a search from here
        neighbors = list(get_neighboring_points(vertex, segment_point_map))
        graph[vertex] = neighbors
        if len(neighbors) > 2:
            print('Found a fork / intersection')
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)

    return graph


def get_segment_point(segment, index):
    return tuple(segment['SplineInfo_Points'][index]['OutVal'])


if __name__ == '__main__':
    main()
