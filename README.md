# Deepdrive sim [![Build Status](https://travis-ci.org/deepdrive/deepdrive-sim.svg?branch=master)](https://travis-ci.org/deepdrive/deepdrive-sim) [![Build status](https://ci.appveyor.com/api/projects/status/84wj7jsxnymi8uxy?svg=true)](https://ci.appveyor.com/project/crizCraig/deepdrive-sim)


Unreal based simulator and Python interface to Deepdrive


## Usage

Checkout our [main repo](https://github.com/deepdrive/deepdrive)

## Development

- Clone this repo
- Clone our [UnrealEnginePython fork](https://github.com/deepdrive/UnrealEnginePython) into the root of this project (not a submodule as we are going to make this a binary only)

```
cd Plugins
git clone https://github.com/deepdrive/UnrealEnginePython
```

- Tip: To avoid rebuilding UnrealEnginePython, move `Plugins/UnrealEnginePython` to your Engine plugins after you've built it in side `deepdrive-sim/Plugins`.

- [Associate your GitHub username with your Unreal account](https://www.unrealengine.com/en-US/ue4-on-github) to get access to Unreal sources on GitHub. 

### Set your python bin

This is where the Python extension is installed which enables communication between Unreal and Python.

#### Option 1: Automatically set the python bin 

Setup the [deepdrive](https://github.com/deepdrive/deepdrive) project which creates your `~/.deepdrive/python_bin`

#### Option 2: Manually set the python bin 

Create `~/.deepdrive/python_bin` on Unix or `%HOMEPATH%\.deepdrive\python_bin` on Windows and point it to the Python executable you want the deepdrive Python extension to be installed to, i.e. 

```
/home/[YOU]/.local/share/virtualenvs/my-env/bin/python
```

### Windows Development

- Use Visual Studio 2015 with the [C++ build tools](https://stackoverflow.com/a/31955339)
- Get Unreal v4.21 via the Epic Launcher -> Unreal Engine tab -> Library
- Optionally download Unreal sources and debugging symbols in the Epic Launcher

![Windows Unreal Install Options](https://i.imgur.com/Khxc6HV.jpg)

- Download UnrealEnginePython [python binaries] into `Plugins/UnrealEnginePython/Binaries`
- Install the Substance Plugin through the Marketplace in Epic Launcher
- Open DeepDrive.uproject with the version of Unreal Editor you just installed - if there are errors, check `Saved/Logs` for details
- Refresh / Create Visual Studio project
- Open the Visual Studio Project
- Close Unreal
- Right click the DeepDrive project and set as Startup Project, debug...
- To run game full speed when the window is not focused
  - Uncheck Edit->Editor Preferences->Use Less CPU when in Background


### Linux Development

##### Clone Unreal

```
git clone git@github.com:EpicGames/UnrealEngine --branch 4.21
# or if you are using http: 
# git clone https://github.com/EpicGames/UnrealEngine --branch 4.21
```

##### Build Unreal

_Takes an hour with 12 cores_

```
cd UnrealEngine
./Setup.sh && ./GenerateProjectFiles.sh && make
```

More details on building Unreal [here](https://wiki.unrealengine.com/Building_On_Linux), though the above commands should be sufficient.

##### Get the substance plugin

Download the Substance plugin from [here](https://forum.allegorithmic.com/index.php/topic,26732.0.html) to
<kbd>UnrealEngine/Plugins/Runtime</kbd> for Unreal 4.21. For other releases, see [here](https://forum.allegorithmic.com/index.php/board,23.0.html) or you can just use the sources downloaded by Windows / Mac marketplace.

##### Run the editor
```
./Engine/Binaries/Linux/UE4Editor
```

Open the Deepdrive uproject file - choose other -> Skip Conversion

## Building the Python Extension

Note that this will happen automatically when you build the Unreal project, so is only needed if you are building this separately.

```
cd Plugins/DeepDrivePlugin/Source/DeepDrivePython
python build/build.py --type dev
```


## Push PyPi module

Pushing to the release branch causes CI to publish wheels to the PyPi cheese shop.

`git push origin master && git push origin master:release`

## Setting key binds

Unreal->Project Settings->Input->Action Mappings OR in Blueprints->Find (uncheck Find in Current Blueprint Only) and search for the input key, i.e. J.

## Commit your VERSION file

Used for checking compatibility between sim and agents and will update automatically after building in Unreal. Please check it in.

## Compiling changes to plugins

If you make a change to the DeepDrivePlugin, you'll often have to recompile the module to see the changes.
To do so in Unreal Editor, open Window->Developer Tools->Modules and search for DeepDrivePlugin.
Then hit Recompile to compile your plugin changes.

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


#### Vulkan

With Unreal 4.21, Vulkan is automatically supported. I've tried it with NVIDIA 384 drivers and things crash, but newer drivers may work.

If you experience crashes and see mentions to Vulkan in the logs, you can ensure OpenGL is used by uinstalling these debian packages

```
sudo apt remove libvulkan1 mesa-vulkan-drivers vulkan-utils
```