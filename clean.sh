#!/usr/bin/env bash

set -e


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"


echo Cleaning derivative data from ${DIR}

set -v
rm -rf ${DIR}/DerivedDataCache
rm -rf ${DIR}/Intermediate
rm -rf ${DIR}/Binaries
rm -rf ${DIR}/Plugins/DeepDrivePlugin/Binaries
rm -rf ${DIR}/Plugins/DeepDrivePlugin/Intermediate
set +v

echo Clean as a whistle!
