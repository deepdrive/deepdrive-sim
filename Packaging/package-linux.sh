#!/usr/bin/env bash
ROOTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && cd .. && pwd )"
docker run --rm -ti "-v$ROOTDIR:/hostdir" -w /hostdir adamrehn/ue4-full:4.21.1 python3 /hostdir/Packaging/package.py
