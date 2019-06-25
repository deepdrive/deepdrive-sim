#!/usr/bin/env bash

set -e  # Abort script at first error, when a command exits with non-zero status (except in until or while loops, if-tests, list constructs)
set -u  # Attempt to use undefined variable outputs error message, and forces an exit
#set -x  # Similar to verbose mode (-v), but expands commands
set -o pipefail  # Causes a pipeline to return the exit status of the last command in the pipe that returned a non-zero return value.

cppcheck --force --enable=all . |& tee -a cppcheck_all.txt
cppcheck --force . 2> cppcheck_err.txt
cat cppcheck_err.txt

if [[ -s cppcheck_err.txt ]]; then
    # There were errors detected by cppcheck, fail
    exit 1
else
    # delete if 0 bytes to avoid s3 upload errors
    rm cppcheck_err.txt
fi