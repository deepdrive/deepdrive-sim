#!/usr/bin/env bash

# Usage: DEEPDRIVE_UNREAL_SOURCE_DIR=YourSourceDir DEEPDRIVE_PACKAGE_DIR=YourPackageDir ./package.sh
# Or with Jenkins
# docker run -u root -d -p 8080:8080 -p 50000:50000 -e DEEPDRIVE_UNREAL_SOURCE_DIR='YourSourceDir' -e DEEPDRIVE_PACKAGE_DIR='YourPackageDir' -v YourJenkinsDir:/var/jenkins_home -v /var/run/docker.sock:/var/run/docker.sock jenkinsci/blueocean

set -euvo pipefail

dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
root_dir="$(dirname "$dir")"
echo root dir is ${root_dir}
unreal_dir=${DEEPDRIVE_UNREAL_SOURCE_DIR}
output_dir=${DEEPDRIVE_PACKAGE_DIR}
user=${DEEPDRIVE_USER}

# This needs to change if you are not clearing /Intermediate files in the project every build.
internal_build_id=8861

${root_dir}/clean.sh

cd ${unreal_dir}/Engine/Build/BatchFiles

sudo chown -Rh ${user}:${user} ${root_dir}

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