#!/usr/bin/env bash

set -euxo pipefail

# TODO: Do this in Python (mostly done in package_windows.py already)

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PKG_DIR=${HOME}/deepdrive-packaged/LinuxNoEditor  # TODO: Move this to /var or somewhere

mv ${PKG_DIR}/DeepDrive/Binaries/Linux/DeepDrive-Linux-Shipping ${PKG_DIR}/DeepDrive/Binaries/Linux/DeepDrive &> /dev/null || :
chmod +x ${PKG_DIR}/DeepDrive/Binaries/Linux/DeepDrive

version=`python get_package_version.py`

cd ${PKG_DIR}
file_name=deepdrive-sim-linux-"$version".zip
file_path=/tmp/"$file_name"

rm "$file_path"
zip -r "$file_path" *
cd -

aws s3 cp "$file_path" s3://deepdrive/sim/"$file_name"
rm "$file_path"