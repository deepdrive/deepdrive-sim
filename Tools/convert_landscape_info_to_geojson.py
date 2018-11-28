import json


def main():
    with open('landscape_segments.json') as f:
        segments = json.load(f)

    # Segments seem already de-duped, but just in case
    segments = get_unique_segments(segments)

    segment_point_map = dict()
    for s in segments:
        add_to_map(s, segment_point_map, _get_segment_outval(0, s))
        add_to_map(s, segment_point_map, _get_segment_outval(1, s))

    splines = []

    while segments:
        # Find overlapping segments using OutVal of control point
        # String mid points together    unique_segments = dict()
        s = segments.pop()
        connected_segs = string_together_segments(
            start_segment=s,
            segment_point_map=segment_point_map,
            segments=segments)

        splines.append(connected_segs)

    print('Num splines', len(splines))


def add_to_map(segment, segment_point_map, point):
    if point in segment_point_map:
        if len(segment_point_map[point]) > 2:
            raise NotImplementedError('Road forks not yet implemented')
        segment_point_map[point].append(segment)
    else:
        segment_point_map[point] = [segment]


def get_unique_segments(segments):
    unique_segments = dict()
    for segment in segments:
        unique_segments[segment['full_name']] = segment
    segments = list(unique_segments.values())
    return segments


def get_next_segment_end_point(segment, point, segment_point_map):
    segments = segment_point_map[point]
    for s in segments:
        if segment != s:
            for pt in [_get_segment_outval(0, s), _get_segment_outval(1, s)]:
                if pt != point:
                    return s, pt
            else:
                raise RuntimeError('Could not match segment')
    else:
        print('Only one segment with this control point')
        return None, None


def string_together_segments(start_segment, segment_point_map, segments):
    connected_segments = [start_segment]
    p = _get_segment_outval(1, start_segment)
    s = start_segment
    while True:
        s, p = get_next_segment_end_point(s, p, segment_point_map)
        if s is None or s == start_segment:
            break
        connected_segments.append(s)
        if s not in segments:
            print('Found an inner loop')
            break
        else:
            segments.remove(s)

    return connected_segments


def _get_segment_outval(index, segment):
    return tuple(segment['SplineInfo_Points'][index]['OutVal'])


if __name__ == '__main__':
    main()
