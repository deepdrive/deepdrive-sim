#!/usr/bin/env bash

# c.f. https://simdocs.deepdrive.io/v/v3/docs/setup/linux/ue4docker-export

./clean_all.sh
ue4 build
ue4 run