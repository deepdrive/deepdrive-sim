#!/usr/bin/env bash
ROOTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"
docker run --net=host -ti "-v$ROOTDIR:/hostdir" -v ~/.gcpcreds:/.gcpcreds  -e DEEPDRIVE_PACKAGE_NO_UPLOAD=1 -e GOOGLE_APPLICATION_CREDENTIALS="/.gcpcreds/VoyageProject-d33af8724280.json" -w /hostdir deepdriveio/ue4-deepdrive-deps:latest /hostdir/Packaging/in-docker-package.sh
