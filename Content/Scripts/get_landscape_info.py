import os

import unreal_engine as ue
from unreal_engine.classes import Landscape

import api_methods
import json


def main():
    """Create 3 spline components (L,R,Center) from landscape spline components"""
    for world in ue.all_worlds():
        print('getting world ' + str(world))
        get_landscape_info(world)

    world = api_methods.get_world() #ue.get_editor_world()
    get_landscape_info(world)


def get_landscape_info(world):
    print('Getting landscape info')
    try:
        objects = world.all_objects()
    except Exception:
        print('Could not get objects, trying editor world')
        world = ue.get_editor_world()
        objects = world.all_objects()
    ctrl_pts = []
    segments = []
    utils_proxy = None
    for o in objects:
        full_name = o.get_full_name()
        try:
            if full_name.startswith('LandscapeSplineControlPoint '):
                ctrl_pts.append(o)
            elif full_name.startswith('LandscapeSplineSegment '):
                # print(full_name)
                segments.append(o)
            elif full_name.startswith('UtilsProxy '):
                utils_proxy = o
                # print(o.get_full_name())
                # print(dir(o))
                # print('bCanEverTick ' + str(o.PrimaryActorTick.bCanEverTick))
                # print('GetSplineSegmentConnectionsASDF ' + str(o.GetSplineSegmentConnections()))
                # print('levels ' + str(o.get_levels()))
        except:
            print('error getting object')


    print_segments(utils_proxy, segments)

    print_ctrl_points(ctrl_pts)


def print_ctrl_points(ctrl_pts):
    out_pts = []
    for p in ctrl_pts:
        out_pt = serialize_ctrl_point(p)
        out_pts.append(out_pt)
        print('ctrl point location ' + str(p.Location))
        print('ctrl point rotation ' + str(p.Rotation))
        print(p.SideFalloff)
        print(p.SegmentMeshOffset)
        print('pt.Width ' + str(p.Width))
        print('pt info ' + str(p))
        print('pt full name ' + str(p.get_full_name()))
        print('pt connected segments ' + str(p.ConnectedSegments))
        print('len pt connected segments ' + str(len(p.ConnectedSegments)))
        print(len(p.Points))
        if len(p.Points) > 0:
            print('point center ' + str(p.Points[0].Center))
        # print('length of interp points %d' % len(pt.GetPoints()))
    print('num spline ctrl points ' + str(len(ctrl_pts)))

    with open('landscape_ctrl_points.json', 'w') as outfile:
        json.dump(out_pts, outfile)
        print('Saved ctrl points to %s' % os.path.realpath(outfile.name))


def serialize_ctrl_point(p):
    out_pt = {
        'full_name': p.get_full_name(),
        'display_name': p.get_display_name(),
        'as_string': str(p),
        'Location': get_3d_vector(p.Location),
        'Rotation': {'roll': p.Rotation.roll, 'pitch': p.Rotation.pitch, 'yaw': p.Rotation.yaw},
        'SideFalloff': p.SideFalloff,
        'SegmentMeshOffset': p.SegmentMeshOffset,
    }
    return out_pt


def print_segments(utils_proxy, segments):
    out_segs = []
    unique_mid_points = set()
    for i, segment in enumerate(segments):
        end_points = []
        mid_points = []
        conns = segment.Connections
        out_seg = {
            'full_name': segment.get_full_name(),
            'as_string': str(segment),
            'display_name': segment.get_display_name(),
            'layer_name': str(segment.LayerName),
            'Points': mid_points,
            'SplineInfo_Points': end_points,
            'first_Connection': {
                'as_string': str(segment.Connections),
                'ControlPoint': serialize_ctrl_point(conns.ControlPoint),
                'TangentLen': conns.TangentLen,
            }

        }
        points = segment.Points
        spline_info = segment.SplineInfo
        print('spline_info ' + str(spline_info))
        print('spline_info points' + str(spline_info.Points))
        for p in spline_info.Points:

            end_point = {
                'InVal': p.InVal,
                'OutVal': get_3d_vector(p.OutVal),  # Cross-ref this with ControlPoint Location
                'ArriveTangent': get_3d_vector(p.ArriveTangent),
                'LeaveTangent': get_3d_vector(p.LeaveTangent),
                'InterpMode': p.InterpMode,
            }
            # InterpMode 3 CIM_CurveUser =>	/** A smooth curve just like CIM_Curve,
            # but tangents are not automatically updated so you can have manual
            # control over them (eg. in Curve Editor). */

            end_points.append(end_point)

            print(str(i) + ' interp_point_InVal ' + str(p.InVal))
            print(str(i) + ' interp_point_OutVal ' + str(p.OutVal))
            print(str(i) + ' interp_point_ArriveTangent ' + str(p.ArriveTangent))
            print(str(i) + ' interp_point_LeaveTangent ' + str(p.LeaveTangent))
            print('interp_point_InterpMode ' + str(p.InterpMode))
        print('len_spline_info points ' + str(len(spline_info.Points)))
        print('points ' + str(points))
        for p in points:
            mid_point = {
                'Center': get_3d_vector(p.Center),
                'Right': get_3d_vector(p.Right),
                'Left': get_3d_vector(p.Left),
                'FalloffLeft': get_3d_vector(p.FalloffLeft),
                'FalloffRight': get_3d_vector(p.FalloffRight),
                'StartEndFalloff': p.StartEndFalloff
            }
            unique_mid_points.add(tuple(get_3d_vector(p.Center)))
            mid_points.append(mid_point)

        out_segs.append(out_seg)
        print(str(i) + ' point centers ' + str([str(p.Center) for p in points]))
        print('len_points ' + str(len(points)))
        print('connections ' + str(conns))
        conns_str = str(conns)
        conn_ptr = int(conns_str[conns_str.rindex(':')+2:-2], 16)
        # TODO: Get size (currently 24)
        # TODO: Hope that the pointer can be cast
        # TODO: Then get all connections in C++ with GetSplineSegmentConnections
        print('connections ptr ' + str(conn_ptr))
        print('sig ' + str(conns.ref.__text_signature__))
        print('ref ' + str(conns.ref()))
        print('ref dict' + str(conns.ref().as_dict()))
        print('ref fields' + str(conns.ref().fields()))
        print('conns dir' + str(dir(conns)))
        print('ref dir' + str(dir(conns.ref())))
        print('dict' + str(conns.as_dict()))
        print(conns.fields())
        print(conns.ControlPoint)
        print(conns.TangentLen)
        print('conns.get_field_array_dim(\'ControlPoint\'))' + str(conns.get_field_array_dim('ControlPoint')))
        print('seg call_function ' + str(segment.call_function))
        if utils_proxy:
            print('utils call_function ' + str(utils_proxy.call_function))
        # print('segment uobject ' + str(segment.get_owner()))
        # print(utils_proxy.GetSplineSegmentConnections(conn_ptr))
            print('GetSplineSegmentConnections ' + str(utils_proxy.GetSplineSegmentConnections()))
        # print('lesgo2 ' + str(utils_proxy.call_function('GetSplineSegmentConnections')))
    print('num mid points ' + str(len(unique_mid_points)))

    with open('landscape_segments.json', 'w') as outfile:
        json.dump(out_segs, outfile)
        print('Saved segments to %s' % os.path.realpath(outfile.name))


    print('Num segments ' + str(len(segments)))


def get_3d_vector(p):
    return [p.x, p.y, p.z]


if __name__ == '__main__':
    main()
