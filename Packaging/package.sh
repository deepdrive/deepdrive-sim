#!/usr/bin/env bash

# Usage: DEEPDRIVE_UNREAL_SOURCE_DIR=YourSourceDir DEEPDRIVE_PACKAGE_DIR=YourPackageDir ./package.sh
# Or with Jenkins
# docker run -u root -d -p 8080:8080 -p 50000:50000 -e DEEPDRIVE_UNREAL_SOURCE_DIR='YourSourceDir' -e DEEPDRIVE_PACKAGE_DIR='YourPackageDir' -v YourJenkinsDir:/var/jenkins_home -v /var/run/docker.sock:/var/run/docker.sock jenkinsci/blueocean

set -euvo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$DIR")"
UNREAL_DIR=${DEEPDRIVE_UNREAL_SOURCE_DIR}
OUTPUT_DIR=${DEEPDRIVE_PACKAGE_DIR}

cd ${UNREAL_DIR}/Engine/Build/BatchFiles

./RunUAT.sh -ScriptsForProject=${ROOT_DIR}/DeepDrive.uproject BuildCookRun \
    -nocompileeditor -nop4 -project=${ROOT_DIR}/DeepDrive.uproject -cook -stage -archive \
    -archivedirectory=${OUTPUT_DIR} -package -clientconfig=Development -ue4exe=UE4Editor -clean -pak -prereqs \
    -nodebuginfo -targetplatform=Linux -build