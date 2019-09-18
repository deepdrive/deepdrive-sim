# Packaging

## Setup

```
 pip install -r Packaging/requirements.txt
```

## Linux

To package a new binary version in Linux run
```
Packaging/package-linux.sh
```

NOTE: Uploads to S3 are done automatically through CI after tests have passed.
To upload to S3 anyway:

```
python Packaging/package.py --upload-only
```



## Windows

In Windows we go through the Unreal Editor packaging interface, and manually
add UnrealEnginePython. TODO: Fill this out. 
