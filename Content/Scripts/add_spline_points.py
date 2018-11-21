import unreal_engine as ue
from unreal_engine.classes import SplineComponent



def main():
    # new_landscape = ue.get_editor_world().actor_spawn(SplineComponent)
    spline = ue.find_class('SplineComponent')
    print(spline.AddSplinePoint)
    # print(dir(spline))


if __name__ == '__main__':
    main()