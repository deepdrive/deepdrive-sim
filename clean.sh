#!/usr/bin/env bash

set -e
set -v

echo Cleaning derivative data

rm -rf DerivedDataCache
rm -rf Saved
rm -rf Intermediate
rm -rf Binaries
rm -rf Plugins/DeepDrivePlugin/Binaries
rm -rf Plugins/DeepDrivePlugin/Intermediate
