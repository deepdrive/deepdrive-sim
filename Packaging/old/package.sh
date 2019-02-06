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

rm -rf ${output_dir}/*

${root_dir}/clean.sh

cd ${unreal_dir}/Engine/Build/BatchFiles

sudo chown -Rh ${user}:${user} ${root_dir}

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

# Download python libs
TMPFILE=`mktemp`
wget https://s3-us-west-1.amazonaws.com/deepdrive/unreal_python_lib/python_libs.zip -O ${TMPFILE}
unzip -d python_libs ${TMPFILE}
mv python_libs ${output_dir}/LinuxNoEditor/DeepDrive/
rm ${TMPFILE}

sudo chown -Rh ${user}:${user} ${output_dir}

