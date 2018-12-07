import copy
import json
import collections
import os
import uuid
import numpy as np
from scipy import spatial
from scipy.interpolate import interp1d, BPoly




GAP_CM = 100
USE_TANGENTS = False
CANYONS_FILENAME = 'landscape_segments.json'
MESA_FILENAME = 'landscape_segments_mesa.json'
# LANDSCAPE_OFFSET = np.array([-11200.000000, -11200.000000, 100.000000])  # Mesa offset
LANDSCAPE_OFFSET = np.array([-21214.687500, -21041.564453, 18179.121094])  # Canyons TODO: Get this from Unreal
LANDSCAPE_Z_SCALE = np.array([1, 1, 159.939941 / 100])
INTERSECTIONS = False

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


def get_graphs_by_side_world(graphs_by_side, landscape_offset, landscape_z_scale):
    ret = {}
    for side in graphs_by_side:
        graph = graphs_by_side[side]
        new_graph = {}
        for p in graph:
            p_world = landscape_to_world(landscape_offset, landscape_z_scale, p)
            new_connections = []
            for conn in graph[p]:
                conn_world = landscape_to_world(landscape_offset, landscape_z_scale, conn)
                new_connections.append(tuple(conn_world))
            new_graph[tuple(p_world)] = new_connections
        ret[side] = new_graph
    return ret


def add_point_relations(lane):
    n = len(lane)
    if n < 3:
        return
    for idx, segment in enumerate(lane):

        A = np.array(lane[idx-1]['drive'])
        B = np.array(segment['drive'])
        C = np.array(lane[(idx+1) % n]['drive'])
        a = np.linalg.norm(C - B)
        b = np.linalg.norm(C - A)
        c = np.linalg.norm(B - A)
        s = (a + b + c) / 2
        R = a * b * c / 4 / np.sqrt(s * (s - a) * (s - b) * (s - c))
        print('radius %r' % R)
        segment['radius'] = R
        v = B - A
        segment['direction'] = np.arctan(v[1] / v[0])
        segment['distance'] = c
        print('distance %r' % c)
        if c > 500:
            raise ValueError('Distance too large %r, A: %r B: %r' % (c, A, B))


def add_ids(inner_lane, outer_lane):
    if INTERSECTIONS:
        raise NotImplementedError('Intersections not supported')
    if len(inner_lane) != len(outer_lane):
        raise ValueError('Expected inner and outer lane lengths too match')

    for i, inner_seg in enumerate(inner_lane):
        outer_seg = outer_lane[i]
        inner_seg_id = new_seg_id()
        outer_seg_id = new_seg_id()
        add_seg_ids(inner_seg, inner_seg_id, outer_seg_id)
        add_seg_ids(outer_seg, outer_seg_id, inner_seg_id)

    ret = inner_lane + outer_lane
    for i, seg in enumerate(ret):
        seg['id'] = i
    return ret


def add_seg_ids(seg, my_id, adj_id):
    seg['adjacent_segments'] = [adj_id]
    seg['segment_id'] = my_id


def new_seg_id():
    # 33 billion meters of road - with 1 / billion collision chance =
    # 36**x = 33e18 = 12.54
    ret = str(uuid.uuid4()).replace('-', '')[:13]
    ret = '%s-%s' % (ret[:5], ret[5:])
    return ret


def main():
    with open('landscape_segments.json') as f:
        segments = json.load(f)
    segments = set(HashableSegment(s) for s in segments)
    segment_point_map = get_segment_point_map(segments)
    points = list(segment_point_map.keys())
    graphs = get_map_graphs(points, segment_point_map)
    graphs_by_side = match_graphs_to_side(graphs)

    # The following depends on three separate landscape splines for the road and two guard rails
    side_point_mapping = match_points_to_sides(graphs_by_side, points)
    sides = interpolate_points_by_segment(points, segment_point_map, side_point_mapping,
                                          LANDSCAPE_OFFSET, LANDSCAPE_Z_SCALE)

    graphs_by_side_world = get_graphs_by_side_world(graphs_by_side, LANDSCAPE_OFFSET, LANDSCAPE_Z_SCALE)

    inner_lane, outer_lane = get_lanes(sides, graphs_by_side_world)
    # TODO: Go along drive paths and compute curvature with three consecutive points with 1 / radius, where r is given by https://stackoverflow.com/a/20315063/134077

    add_point_relations(inner_lane)
    add_point_relations(outer_lane)

    all_lane_segments = add_ids(inner_lane, outer_lane)

    with open('deepdrive-canyons-map.json', 'w') as outfile:
        json.dump(serialize_as_geojson(all_lane_segments), outfile)
        print('Saved map to %s' % os.path.realpath(outfile.name))


def get_closest_point(point, kd_tree):
    distance, index = kd_tree.query(point)
    point = kd_tree.data[index]
    return point, distance


def get_lanes(sides, graphs_by_side):
    inner_lane = []
    outer_lane = []
    start_point = (-27059.478515625, 22658.291015750001, 20553.831906106912)  #  (-5844.791015625, 43699.85546875, 1484.7515869140625)
    end_point = None  # (-8673.3046875, 42718.24609375, 1666.61572265625)
    center = start_point
    prev = (-27128.758053206424, 22593.083229395124, 20568.862235249715)
    prev_drive_inner = None
    prev_drive_outer = None
    inner_kd_tree = spatial.KDTree(sides['inner'])
    outer_kd_tree = spatial.KDTree(sides['outer'])
    center_kd_tree = spatial.KDTree(sides['center'])
    centers = sides['center']
    center_idxs = get_center_idxs(centers)
    while True:
        inner, inner_dist = get_closest_point(center, inner_kd_tree)
        outer, outer_dist = get_closest_point(center, outer_kd_tree)
        check_lane_width(max(inner_dist, outer_dist))
        inner_lane_segment = get_lane(center, inner, prev_drive_inner, is_opposite=False)
        outer_lane_segment = get_lane(center, outer, prev_drive_outer, is_opposite=True)
        center, prev = get_next_center(center, graphs_by_side, prev, center_kd_tree), center
        if end_point is not None and np.array_equal(center, end_point):
            break
        if prev_drive_inner is None or prev_drive_outer is None:
            print('Bootstrapping prev inner and outer')
            end_point = center
        else:
            inner_lane.append(inner_lane_segment)
            outer_lane.append(outer_lane_segment)
        prev_drive_inner = inner_lane_segment['drive']
        prev_drive_outer = outer_lane_segment['drive']
    return inner_lane, outer_lane


def get_center_idxs(centers):
    center_idxs = collections.defaultdict(list)
    for i, p in enumerate(centers):
        center_idxs[tuple(p)].append(i)
    return center_idxs


def get_next_center(center, graphs_by_side, prev, kd_tree):
    seg_end_points = graphs_by_side['center'].get(tuple(center), None)
    if INTERSECTIONS and seg_end_points is not None:
        # TODO: BFS on the interpolated graph segments.

        # Get the next segment. Same as getting next interpolated point when there are no intersections,
        # but I don't want to fail silently if there are intersections.
        if len(seg_end_points) > 2:
            raise ValueError('Intersections and forks not yet supported')
        else:
            return [p for p in graphs_by_side['center'][center] if p != center][0]
    else:
        # Grab the nearest interpolated point not equal to prev
        results = kd_tree.query(np.array(center), 256)
        for i, d in enumerate(results[0]):
            if d == 0:
                continue
            p = kd_tree.data[results[1][i]]
            if not np.array_equal(prev, p):
                return tuple(p)
        else:
            raise ValueError('Could not find next center')


def check_lane_width(dist):
    if dist > 20 * 100:
        raise ValueError('Got lane width way wider than expected')


def get_lane(left, right, prev, is_opposite):
    left = np.array(left)
    drive = left + (right - left) * 0.5
    if prev is None:
        direction = None
    else:
        if is_opposite:
            vector = prev - right
        else:
            vector = right - prev
        direction = np.arctan(vector[1] / vector[0])
    lane = {'left': left, 'drive': drive, 'right': right, 'direction': direction}
    return lane


def interpolate_points_by_segment(points, segment_point_map, side_point_mapping, landscape_offset, landscape_z_scale):
    visited_segments = set()
    sides = collections.defaultdict(list)
    for point in points:
        if point not in side_point_mapping:
            print('Ignoring point %r - not on track and should be deleted' % (point,))
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
            #       For the canyons map, the cubic interpolation looks perfect. For the mesa test map, it doesn't.
            #       So, their could be some design params that can help avoid raycasting everything.

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
    out_segments = []
    for i in range(len(in_segments) - 1):
        start = in_segments[i]
        end = in_segments[i + 1]
        out_segments.append(
            get_geojson_lane_segment(end, start))
    out['lanes'] = {'features': out_segments, 'type': 'FeatureCollection'}
    return out


def get_geojson_lane_segment(segment, next_segment):
    # TODO: Convert all cm values to meters (right, left)
    properties = {
        'vehicle_type': "VEHICLE",
        'intersections': [],
        'lane_count': 1,
        'lane_num': 1,
        'lane_type': "STRAIGHT",
        'next_lanes': [],
        'other_attributes': [],
        'prev_lanes': [],
        'speed_limit': 60,
        'radius': segment['radius'] / 100,
        'distance': segment['distance'] / 100,
        'id': segment['id'],
        'segment_id': segment['segment_id'],
        'adjacent_segments': segment['adjacent_segments'],
    }
    coords = get_geojson_coords(segment, next_segment, 'drive')
    right_coords = get_geojson_coords(segment, next_segment, 'right')
    left_coords = get_geojson_coords(segment, next_segment, 'left')
    properties['right_line'] = get_geojson_line(right_coords)
    properties['left_line'] = get_geojson_line(left_coords)
    polygon = np.array([segment['left'], next_segment['left'], segment['right'], next_segment['right']]) / 100
    properties['polygon'] = {'coordinates': polygon.tolist(), 'type': 'Polygon'}
    ret = {
        'geometry': {'type': 'LineString', 'coordinates': coords},
        'properties': properties,
        'type': 'Feature'}
    return ret


def get_geojson_line(left_coords):
    return {'geometry': {'type': 'LineString', 'coordinates': left_coords}, 'line_color': 'WHITE',
            'line_type': 'DOUBLE_SOLID_LINE'}


def get_geojson_coords(segment, next_segment, side):
    return (np.array([segment[side], next_segment[side]]) / 100).tolist()


"""
geometry: {type: "LineString", coordinates: [[-91.23195947978866, 115.201464914047, 125.591217041],…]}
coordinates: [[-91.23195947978866, 115.201464914047, 125.591217041],…]
0: [-91.23195947978866, 115.201464914047, 125.591217041]
1: [-91.17320342247241, 114.30339182956251, 125.601303101]
type: "LineString"
line_color: "NONE"
line_type: "IMPLIED"
"""


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
    :param landscape_z_scale: Z-scale of landscape in Unreal transform
    :param segment: Unreal segment
    :param side: Left, Center, or Right
    :return:
    """

    orig = get_side_points(landscape_offset, landscape_z_scale, segment, side)
    if len(orig) <= 2:
        return np.array(orig)
    else:
        # Do cubic interpolation to smooth out inner points
        if len(orig) < 4:
            points = add_midpoints(orig)
        else:
            points = orig
        distances = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        distances = np.hstack([[0], distances])
        points = np.array(points)
        total_points = distances[-1] // GAP_CM + 1
        chords = list(np.arange(0, total_points) * GAP_CM)
        if distances[-1] % GAP_CM > (GAP_CM / 3):
            # Add an extra interp point so there won't be big gap
            chords.append(distances[-1])
        cubic_interpolator = interp1d(distances, points, 'cubic', axis=0)
        ret = cubic_interpolator(chords)
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
