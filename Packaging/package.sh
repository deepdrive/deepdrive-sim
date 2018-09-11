#!/usr/bin/env bash

# Usage: DEEPDRIVE_UNREAL_SOURCE_DIR=YourSourceDir DEEPDRIVE_PACKAGE_DIR=YourPackageDir ./package.sh
# Or with Jenkins
# Add the following to /etc/environment and restart
# DEEPDRIVE_PACKAGE_DIR="<some-dir>"
# DEEPDRIVE_UNREAL_SOURCE_DIR="<your-unreal-repo>"
# DEEPDRIVE_USER="<your-username>"
set -euvo pipefail

dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
root_dir="$(dirname "$dir")"
echo root dir is ${root_dir}
unreal_dir=${DEEPDRIVE_UNREAL_SOURCE_DIR}
output_dir=${DEEPDRIVE_PACKAGE_DIR}
user=${DEEPDRIVE_USER}

echo DEEPDRIVE_UNREAL_SOURCE_DIR ${DEEPDRIVE_UNREAL_SOURCE_DIR}
echo DEEPDRIVE_PACKAGE_DIR ${DEEPDRIVE_PACKAGE_DIR}
echo DEEPDRIVE_USER ${DEEPDRIVE_USER}

${root_dir}/clean.sh

cd ${unreal_dir}/Engine/Build/BatchFiles

sudo chown -Rh ${user}:${user} ${root_dir}

# Development_Server

# 	<PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Development_Server|x64'">
#		<LocalDebuggerCommandArguments>"$(SolutionDir)$(ProjectName).uproject" -skipcompile</LocalDebuggerCommandArguments>
#		<DebuggerFlavor>WindowsLocalDebugger</DebuggerFlavor>


#	<PropertyGroup Condition="'$(Configuration)|$(Platform)'=='Development_Server|x64'">
#		<IncludePath />
#		<ReferencePath />
#		<LibraryPath />
#		<LibraryWPath />
#		<SourcePath />
#		<ExcludePath />
#		<OutDir>$(ProjectDir)..\Build\Unused\</OutDir>
#		<IntDir>$(ProjectDir)..\Build\Unused\</IntDir>
#		<NMakeBuildCommandLine>C:\Users\a\src\UnrealEngine\Engine\Build\BatchFiles\Build.bat DeepDriveServer Win64 Development "$(SolutionDir)$(ProjectName).uproject" -waitmutex</NMakeBuildCommandLine>
#		<NMakeReBuildCommandLine>C:\Users\a\src\UnrealEngine\Engine\Build\BatchFiles\Rebuild.bat DeepDriveServer Win64 Development "$(SolutionDir)$(ProjectName).uproject" -waitmutex</NMakeReBuildCommandLine>
#		<NMakeCleanCommandLine>C:\Users\a\src\UnrealEngine\Engine\Build\BatchFiles\Clean.bat DeepDriveServer Win64 Development "$(SolutionDir)$(ProjectName).uproject" -waitmutex</NMakeCleanCommandLine>
#		<NMakeOutput>..\..\Binaries\Win64\DeepDriveServer.exe</NMakeOutput>
#	</PropertyGroup>

#		<NMakeBuildCommandLine>C:\Users\a\src\UnrealEngine\Engine\Build\BatchFiles\Build.bat DeepDriveServer Linux Development "$(SolutionDir)$(ProjectName).uproject" -waitmutex</NMakeBuildCommandLine>
#		<NMakeReBuildCommandLine>C:\Users\a\src\UnrealEngine\Engine\Build\BatchFiles\Rebuild.bat DeepDriveServer Linux Development "$(SolutionDir)$(ProjectName).uproject" -waitmutex</NMakeReBuildCommandLine>
#		<NMakeCleanCommandLine>C:\Users\a\src\UnrealEngine\Engine\Build\BatchFiles\Clean.bat DeepDriveServer Linux Development "$(SolutionDir)$(ProjectName).uproject" -waitmutex</NMakeCleanCommandLine>

@echo off
setlocal enabledelayedexpansion

#  REM The %~dp0 specifier resolves to the path to the directory where this .bat is located in.
#  REM We use this so that regardless of where the .bat file was executed from, we can change to
#  REM directory relative to where we know the .bat is stored.
#  pushd "%~dp0\..\..\Source"
#
#  REM %1 is the game name
#  REM %2 is the platform name
#  REM %3 is the configuration name
#
#  IF EXIST ..\..\Engine\Binaries\DotNET\UnrealBuildTool.exe (
#          ..\..\Engine\Binaries\DotNET\UnrealBuildTool.exe %* -DEPLOY
#  		popd
#
#  		REM Ignore exit codes of 2 ("ECompilationResult.UpToDate") from UBT; it's not a failure.
#  		if "!ERRORLEVEL!"=="2" (
#  			EXIT /B 0
#  		)
#
#  		EXIT /B !ERRORLEVEL!
#  ) ELSE (
#  	ECHO UnrealBuildTool.exe not found in ..\..\Engine\Binaries\DotNET\UnrealBuildTool.exe
#  	popd
#  	EXIT /B 999
#  )

# Server
# C:\Users\a\src\UnrealEngine\Engine\Build\BatchFiles\Build.bat DeepDriveServer Linux Development "C:\Users\a\src\deepdrive-sim\DeepDrive.uproject" -waitmutex

# Build project including Deepdrive and DeepdrivePlugin modules - TODO: Use Build.sh?
sudo -u ${user} HOME=/home/${user} ${unreal_dir}/Engine/Binaries/DotNET/UnrealBuildTool.exe DeepDrive Development \
    Linux -project="${root_dir}/DeepDrive.uproject" -editorrecompile -progress -NoHotReloadFromIDE

# Package
sudo -u ${user} HOME=/home/${user} ./RunUAT.sh -ScriptsForProject=${root_dir}/DeepDrive.uproject BuildCookRun \
    -nocompileeditor -nop4 -project=${root_dir}/DeepDrive.uproject -cook -stage -archive \
    -archivedirectory=${output_dir} -package -clientconfig=Development -ue4exe=UE4Editor -clean -pak -prereqs \
    -nodebuginfo -targetplatform=Linux -build

# npm install --global empty-trash-cli
# empty-trash??????