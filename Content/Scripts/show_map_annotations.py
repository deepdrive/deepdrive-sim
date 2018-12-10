import json
try:
    import unreal_engine as ue
    from unreal_engine.classes import Blueprint
    from unreal_engine import FVector, FRotator
    DRY_RUN = False
    WORLD = ue.get_editor_world()
except ImportError:
    DRY_RUN = True


with open(r'C:\Users\a\src\deepdrive-sim\Tools\deepdrive-canyons-map.json') as f:
    geojson = json.load(f)

def spawn_point_marker(point, map_point_class):
    if not DRY_RUN:
        actor = WORLD.actor_spawn(map_point_class,
                                  FVector(*point), FRotator(0, 0, 0))

def to_unreal(point, lift_cm):
    return point[0] * 100, point[1] * 100, point[2] * 100 + lift_cm


def main():
    """
    Output https://youtu.be/kqfTDl2p9ts
    """
    if DRY_RUN:
        right_orb = None
        left_orb = None
        center_orb = None
    else:
        right_orb = ue.load_object(Blueprint, '/Game/DeepDrive/Blueprints/MapPointRight').GeneratedClass
        left_orb = ue.load_object(Blueprint, '/Game/DeepDrive/Blueprints/MapPointLeft').GeneratedClass
        center_orb = ue.load_object(Blueprint, '/Game/DeepDrive/Blueprints/MapPoint').GeneratedClass
    for i, lane_seg in enumerate(geojson['lanes']['features']):
        center = lane_seg['geometry']['coordinates']
        left = lane_seg['properties']['left_line']['geometry']['coordinates']
        right = lane_seg['properties']['right_line']['geometry']['coordinates']

        spawn_point_marker(to_unreal(left[0], 0), left_orb)
        spawn_point_marker(to_unreal(left[1], 0), left_orb)
        spawn_point_marker(to_unreal(right[0], 60), right_orb)
        spawn_point_marker(to_unreal(right[1], 60), right_orb)
        spawn_point_marker(to_unreal(center[0], 30), center_orb)
        spawn_point_marker(to_unreal(center[1], 30), center_orb)


if __name__ == '__main__':
    main()
