#!/usr/bin/env bash

set -e


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


echo Cleaning derivative data from ${DIR}

set -v
rm -rf ${DIR}/DerivedDataCache
rm -rf ${DIR}/Intermediate
rm -rf ${DIR}/Build
rm -rf ${DIR}/Binaries
rm -rf ${DIR}/Plugins/DeepDrivePlugin/Binaries
rm -rf ${DIR}/Plugins/DeepDrivePlugin/Intermediate
rm -rf ${DIR}/.kdev4
rm -rf ${DIR}/.vscode
rm -rf ${DIR}/CMakeLists.txt
rm -rf ${DIR}/DeepDrive.code-workspace
rm -rf ${DIR}/DeepDrive.kdev4
rm -rf ${DIR}/DeepDrive.pro
rm -rf ${DIR}/DeepDrive.workspace
rm -rf ${DIR}/DeepDriveConfig.pri
rm -rf ${DIR}/DeepDriveDefines.pri
rm -rf ${DIR}/DeepDriveHeader.pri
rm -rf ${DIR}/DeepDriveIncludes.pri
rm -rf ${DIR}/DeepDriveSource.pri
rm -rf ${DIR}/DeepDriveCodeLitePreProcessor.txt
rm -rf ${DIR}/DeepDriveCodeCompletionFolders.txt

# In case this is executed with cygwin, don't delete python binaries - just UEPy ones
find ${DIR}/Plugins/UnrealEnginePython/Binaries -type f -iname '*unreal*' -delete
find ${DIR}/Plugins/UnrealEnginePython/Binaries -type f -iname '*ue4editor*' -delete

rm -rf ${DIR}/Plugins/UnrealEnginePython/Intermediate
set +v

echo Clean as a whistle!
