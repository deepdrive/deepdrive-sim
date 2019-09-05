#!/usr/bin/env bash

set -e  # Abort script at first error, when a command exits with non-zero status (except in until or while loops, if-tests, list constructs)
set -u  # Attempt to use undefined variable outputs error message, and forces an exit
#set -x  # Similar to verbose mode (-v), but expands commands
set -o pipefail  # Causes a pipeline to return the exit status of the last command in the pipe that returned a non-zero return value.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# TODO: Use gsutil rsync for updates

JAMESTOWN_ZIP=${DIR}/JamestownParallelDomainDeepdriveSmallMap.zip
if [[ ! -f "${JAMESTOWN_ZIP}" ]]; then
    wget https://storage.googleapis.com/deepdriveio/unreal_assets/JamestownParallelDomainDeepdriveSmallMap.zip ${DIR}
    # mmd5sum 6ce83f6a9a359bd83ba1b22dbf71c4a4
fi
unzip ${JAMESTOWN_ZIP} -d ${DIR}/Plugins/DeepDriveCityPlugin/Content
rm ${JAMESTOWN_ZIP}

# TODO: Kevindale from private bucket
