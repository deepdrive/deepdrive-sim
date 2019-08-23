#!/usr/bin/env bash

zip_name=`ls -la deepdrive-sim-linux*|sort -r|fmt|tail -n 1 | awk 'NF>1{print $NF}'`
~/.local/bin/aws --profile=default s3 cp ${zip_name} s3://deepdrive/sim/${zip_name}
