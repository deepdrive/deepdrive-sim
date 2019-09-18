#!/usr/bin/env bash
ROOTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"
docker run --rm -ti "-v$ROOTDIR:/hostdir" -e DEEPDRIVE_PACKAGE_NO_UPLOAD=1 -w /hostdir deepdriveio/ue4-deepdrive-deps:latest /hostdir/Packaging/in-docker-package.sh
