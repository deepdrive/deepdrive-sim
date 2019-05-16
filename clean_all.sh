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
rm -rf ${DIR}/CMakeLists.txt

# In case this is executed with cygwin, don't delete python binaries - just UEPy ones
find ${DIR}/Plugins/UnrealEnginePython/Binaries -type f -iname '*unreal*' -delete || echo not found
find ${DIR}/Plugins/UnrealEnginePython/Binaries -type f -iname '*ue4editor*' -delete || echo not found

rm -rf ${DIR}/Plugins/UnrealEnginePython/Intermediate || echo not found
set +v

echo Clean as a whistle!
