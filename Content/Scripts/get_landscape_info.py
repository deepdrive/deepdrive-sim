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


    # for o in lspline_ctrl_pts:
    #     print(o.get_full_name())




def get_landscape_info(world):
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
            if full_name.startswith('LandscapeSplineControlPoint '
                                    '/Game/DeepDrive/Maps/Sublevels/DeepDrive_Canyon.'
                                    'DeepDrive_Canyon:PersistentLevel.Landscape4.'
                                    'LandscapeSplinesComponent'):
                ctrl_pts.append(o)
            elif full_name.startswith('LandscapeSplineSegment /Game/DeepDrive/Maps/Sublevels/'
                                      'DeepDrive_Canyon.DeepDrive_Canyon:PersistentLevel.'
                                      'Landscape4.LandscapeSplinesComponent'):
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
    for pt in ctrl_pts:
        # print(dir(pt))
        # print(pt.get_points())
        print('point location ' + str(pt.Location))
        print(pt.SideFalloff)
        print(pt.SegmentMeshOffset)
        print(len(pt.Points))
        if len(pt.Points) > 0:
            print('point center ' + str(pt.Points[0].Center))
        # print('length of interp points %d' % len(pt.GetPoints()))
    print('num spline ctrl points ' + str(len(ctrl_pts)))


def print_segments(utils_proxy, segments):
    for segment in segments:
        conns = segment.Connections
        print('connections ' + str(conns))
        conn_ptr = int(str(conns)[-18:-2], 16)
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
        print('utils call_function ' + str(utils_proxy.call_function))
        # print('segment uobject ' + str(segment.get_owner()))
        # print(utils_proxy.GetSplineSegmentConnections(conn_ptr))
        print('GetSplineSegmentConnections ' + str(utils_proxy.GetSplineSegmentConnections()))
        # print('lesgo2 ' + str(utils_proxy.call_function('GetSplineSegmentConnections')))



    print('Num segments ' + str(len(segments)))


if __name__ == '__main__':
    main()
