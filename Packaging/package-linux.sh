#!/usr/bin/env bash
ROOTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"
docker run --rm -ti "-v$ROOTDIR:/hostdir" -w /hostdir deepdriveio/ue4-deepdrive-deps:latest python3 /hostdir/Packaging/package.py
