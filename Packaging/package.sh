#!/usr/bin/env bash

# Usage: ./package.sh YourUnrealSourceDirectory YourDesiredOutputDirectory

set -euvo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$DIR")"
UNREAL_DIR=$1
OUTPUT_DIR=$2

cd ${UNREAL_DIR}/Engine/Build/BatchFiles

./RunUAT.sh -ScriptsForProject=${ROOT_DIR}/DeepDrive.uproject BuildCookRun \
    -nocompileeditor -nop4 -project=${ROOT_DIR}/DeepDrive.uproject -cook -stage -archive \
    -archivedirectory=${OUTPUT_DIR} -package -clientconfig=Development -ue4exe=UE4Editor -clean -pak -prereqs \
    -nodebuginfo -targetplatform=Linux -build