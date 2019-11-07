import unreal_engine as ue

# [print(w.get_full_name()) for w in ue.all_worlds() ]

for w in ue.all_worlds():
    print('ppWorld')
    print(w.get_full_name())
    print('ppActors')
    [print(a.get_full_name()) for a in w.all_actors()]