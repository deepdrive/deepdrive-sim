# Building the python bindings 

```
cd Plugins/DeepDrivePlugin/Source/DeepDrivePython
python build/build.py --type dev
```

# Pushing bindings to PyPi

Pushing to the release branch causes CI to publish wheels to the PyPi cheese shop.

```
git push origin master && git push origin master:release
```
