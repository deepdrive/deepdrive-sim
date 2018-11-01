# Deepdrive sim [![Build Status](https://travis-ci.org/deepdrive/deepdrive-sim.svg?branch=master)](https://travis-ci.org/deepdrive/deepdrive-sim) [![Build status](https://ci.appveyor.com/api/projects/status/84wj7jsxnymi8uxy?svg=true)](https://ci.appveyor.com/project/crizCraig/deepdrive-sim)


Unreal based simulator and Python interface to Deepdrive


## Usage

Checkout our [main repo](https://github.com/deepdrive/deepdrive)

## Development

- [Associate your GitHub username with your Unreal account](https://www.unrealengine.com/en-US/ue4-on-github)

### Windows

- Clone the repo
- Download and extract our [embedded python](https://s3-us-west-1.amazonaws.com/deepdrive/embedded_python_for_unreal/windows/python_bin_with_libs.zip) into `Plugins/UnrealEnginePython/Binaries/Win64`  
- Use Visual Studio 2015 with the [C++ build tools](https://stackoverflow.com/a/31955339)
- Get Unreal v4.17 via the Epic Launcher -> Unreal Enngine tab -> Library
- Install the Substance Plugin through the Marketplace in Epic Launcher
- Make sure rc.exe is in your PATH, if not follow [these](https://stackoverflow.com/a/14373113/134077) instructions but for Visual Studio 14.0 and x64 instead of Visual Studio 11, x86.
- Open Unreal and use it to open DeepDrive.uproject - if there are errors, check `Saved/Logs` for details
- Refresh / Create Visual Studio project
- On Windows, Right click the DeepDrive project and set as Startup Project, debug...

- Run game full speed when the window is not focused
  - Uncheck Edit->Editor Preferences->Use Less CPU when in Background

Opening the project in Unreal will update the Substance textures to Windows-only versions. 
Once you see these files changed, you can run the following to avoid ever checking them in
```
cd Content/TuningCars/Materials/Hangar/material/Substance
git update-index --assume-unchanged $(git ls-files | tr '\n' ' ')
```



### Linux

Download the Substance plugin from [here](https://forum.allegorithmic.com/index.php/topic,21919.msg87141/highlight,4.18+Released.html#msg87141) to
<kbd>Plugins/Runtime</kbd>.

Build Unreal

```
cd UnrealEngine
./Setup.sh && ./GenerateProjectFiles.sh && make  # Takes about an hour
```

More details on building Unreal [here](https://wiki.unrealengine.com/Building_On_Linux) - though the above commands should be sufficient.

Run 
```
./Engine/Binaries/Linux/UE4Editor
```

Open deepdrive uproject file - choose other -> Skip Conversion

## Building the Python Extension

```
cd Plugins/DeepDrivePlugin/Source/DeepDrivePython
python build/build.py --type dev
```
This will also happen automatically when building the Unreal project.


## Push PyPi module
`git push origin master && git push origin master:release`

## Setting key binds

Unreal->Project Settings->Input->Action Mappings OR in Blueprints->Find (uncheck Find in Current Blueprint Only) and search for the input key, i.e. J.

## Commit your VERSION file

Until Unreal binary uploads are automated, the server and client version will not match unless the VERSION file changes 
are pushed. This because the client version is determined by latest git timestamp and the server version is determined
by the VERSION file. The VERSION file will automatically update when you build, so all that's needed is to push it.
The only time you shouldn't push your VERSION file is when you are just packaging and uploading the sim.

## Clean builds

You'll often want to run `clean.sh` or `clean.bat` after pulling in changes, especially to the plugin as Unreal will spuriously cache old binaries.

## PyCharm
If you open an Unreal project in Pycharm, add Binaries, Build, Content, Intermediate, and Saved to your project’s “Excluded” directories in Project Structure or simply by right clicking and choosing “Mark Directory as” => “Excluded”. Keeping these large binary directories in the project will cause PyCharm to index them. Do the same with these directories (Binaries, Build, Content, Intermediate, and Saved) within any of the Plugins in the Plugins folder.

## Logging

Command line example

```
-LogCmds="global Verbose, LogPython Verbose, LogAnimMontage off, LogDeepDriveAgent VeryVerbose"
```

In DefaultEngine.ini or Engine.ini:

```
[Core.Log]
global=[default verbosity for things not listed later]
[cat]=[level]
foo=verbose break
```

#### Our log categories

    Defined with DEFINE_LOG_CATEGORY macro

```
LogSunSimulationComponentLogSunLightSimulator
LogDeepDriveAgentControllerBase
LogDeepDriveAgentSteeringController
LogDeepDriveAgent
LogDeepDriveSimulationServer
LogDeepDriveConnectionThread
LogDeepDriveAgentLocalAIController
LogDeepDriveAgentSpeedController
LogDeepDriveSimulationCaptureProxy
LogDeepDriveSimulationMessageHandler
LogDeepDriveSimulationServerProxy
LogDeepDriveSimulation
LogDeepDriveSplineTrack
LogSharedMemCaptureMessageBuilder
LogSharedMemCaptureSinkWorkerLogSharedMemCaptureSinkWorker
LogSharedMemCaptureSinkComponent
DeepDriveCaptureProxy
DeepDriveCaptureComponent
LogDeepDriveCapture
LogCaptureBufferPool
LogDeepDriveConnectionListener
LogDeepDriveClientConnection
LogDeepDriveServer
LogDeepDrivePlugin
LogSharedMemoryImpl_Linux
LogPython
```


#### Verbosities

Fatal

    Fatal level logs are always printed to console and log files and crashes even if logging is disabled.

Error

    Error level logs are printed to console and log files. These appear red by default.

Warning

    Warning level logs are printed to console and log files. These appear yellow by default.

Display

    Display level logs are printed to console and log files.

Log

    Log level logs are printed to log files but not to the in-game console. They can still be viewed in editor as they appear via the Output Log window.

Verbose

    Verbose level logs are printed to log files but not the in-game console. This is usually used for detailed logging and debugging.

VeryVerbose

    VeryVerbose level logs are printed to log files but not the in-game console. This is usually used for very detailed logging that would otherwise spam output.

