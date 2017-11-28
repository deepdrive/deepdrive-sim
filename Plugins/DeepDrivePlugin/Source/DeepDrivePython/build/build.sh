#!/usr/bin/env bash

set -euxo pipefail

ROOT=`git rev-parse --show-toplevel`


# Set patch version to git commit time
export DEEPDRIVE_VERSION=`"${ROOT}"/Packaging/get_package_version.sh`
export DEEPDRIVE_BRANCH=`git rev-parse --abbrev-ref HEAD`

echo "DEEPDRIVE_VERSION is $DEEPDRIVE_VERSION"
echo "DEEPDRIVE_BRANCH is DEEPDRIVE_BRANCH"

PRE_CMD="${PRE_CMD:-}"
DOCKER_IMAGE="${DOCKER_IMAGE:-quay.io/pypa/manylinux1_x86_64}"

# Run locally with sudo -E <command-below> referring to .travis.yml for valid DOCKER_IMAGE and PRE_CMD values
docker run --rm -e "DEEPDRIVE_SRC_DIR=/io" \
    -e PYPI_USERNAME \
    -e PYPI_PASSWORD \
    -e DEEPDRIVE_BRANCH \
    -e DEEPDRIVE_VERSION \
    -v ${ROOT}/Plugins/DeepDrivePlugin/Source:/io \
    ${DOCKER_IMAGE} \
    ${PRE_CMD} \
    /io/DeepDrivePython/build/build-wheels.sh