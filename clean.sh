#!/usr/bin/env bash

set -e
set -v

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


echo Cleaning derivative data from ${DIR}

rm -rf ${DIR}/DerivedDataCache
rm -rf ${DIR}/Saved
rm -rf ${DIR}/Intermediate
rm -rf ${DIR}/Binaries
rm -rf ${DIR}/Plugins/DeepDrivePlugin/Binaries
rm -rf ${DIR}/Plugins/DeepDrivePlugin/Intermediate
