# Export from prebuilt docker image <a name="local-linux-export"></a>

## Prerequisites

- NVIDIA Docker 2

Make sure the ue4-ci-helpers package is up to date

```
pip install --upgrade ue4-ci-helpers
pip install ue4-docker
```

## Build the ue4-full image

https://adamrehn.com/docs/ue4-docker/use-cases/linux-installed-builds

## Build the deepdriveio/ue4-deepdrive-deps image

https://github.com/deepdrive/ue4-deepdrive-deps

## Export Unreal to your local filesystem

```
ue4-docker export installed deepdriveio/ue4-deepdrive-deps:latest ~/UnrealInstalled
```

Set the default version of Unreal with the [ue4cli](https://pypi.org/project/ue4cli/)
 
```
ue4 setroot ~/UnrealInstalled
```

## Build and run the project against the exported UnrealEngine

Now you're ready to build and run the project. This is what you'll do any 
time you've pulled in the latest changes from GitHub as well.

Download all maps

```
./download_all_maps.sh
```

### Clean and build

Do the following the first time you export or anytime you pull latest from
deepdrive-sim.

#### Clean your deepdrive-sim project directory to clear any stale build data

```
./clean_all.sh
```

#### Build and run - takes ~5 minutes

```
ue4 build
ue4 run
```
