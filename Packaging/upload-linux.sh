#!/usr/bin/env bash

zip_name=`ls -la deepdrive-sim-linux*|sort|fmt|tail -n 1`
aws --profile=default s3 cp ${zip_name} s3://deepdrive/sim/${zip_name}