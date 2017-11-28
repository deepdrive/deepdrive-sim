#!/usr/bin/env bash

set -euxo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PKG_DIR=${HOME}/deepdrive-packaged/LinuxNoEditor  # TODO: Move this to /var or somewhere

mv ${PKG_DIR}/DeepDrive/Binaries/Linux/DeepDrive-Linux-Shipping ${PKG_DIR}/DeepDrive/Binaries/Linux/DeepDrive &> /dev/null || :
chmod +x ${PKG_DIR}/DeepDrive/Binaries/Linux/DeepDrive

version=`./get_package_version.sh`

cd ${PKG_DIR}
file_name=deepdrive-sim-"$version".zip
file_path=/tmp/"$file_name"

rm "$file_path"
zip -r "$file_path" *
cd -

aws s3 cp "$file_path" s3://deepdrive/sim/"$file_name"
rm "$file_path"