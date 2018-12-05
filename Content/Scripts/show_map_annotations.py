import json
import unreal_engine as ue
from unreal_engine.classes import Blueprint
from unreal_engine import FVector, FRotator

world = ue.get_editor_world()

with open(r'C:\Users\a\src\deepdrive-sim\Tools\deepdrive-canyons-map.json') as f:
    geojson = json.load(f)

map_point_blueprint = ue.load_object(Blueprint, '/Game/DeepDrive/Blueprints/MapPoint')
map_point_class = map_point_blueprint.GeneratedClass


def spawn_point_marker(point):
    actor = world.actor_spawn(map_point_class,
                                 FVector(*point), FRotator(0, 0, 0))

def main():
    for i, lane_seg in enumerate(geojson['lanes']['features']):
        start = lane_seg['geometry']['coordinates'][0]
        end = lane_seg['geometry']['coordinates'][1]
        spawn_point_marker(start)
        spawn_point_marker(end)
        print('spawned start: %r end: %r' % (start, end))
        if i == 100:
            break


if __name__ == '__main__':
    main()
