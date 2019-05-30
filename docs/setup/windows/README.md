
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