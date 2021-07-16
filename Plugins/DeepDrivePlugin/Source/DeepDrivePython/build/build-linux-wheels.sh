#!/usr/bin/env bash

set -e  # Abort script at first error, when a command exits with non-zero status (except in until or while loops, if-tests, list constructs)
set -u  # Attempt to use undefined variable outputs error message, and forces an exit
set -x  # Similar to verbose mode (-v), but expands commands
set -o pipefail  # Causes a pipeline to return the exit status of the last command in the pipe that returned a non-zero return value.

# Executed within the centos manylinux container

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export DEEPDRIVE_SRC_DIR="$( dirname "$( dirname ${DIR})" )"
echo DEEPDRIVE_SRC_DIR=${DEEPDRIVE_SRC_DIR}

# Get python versions on docker image, i.e. /opt/python/cp35-cp35m/bin /opt/python/cp36-cp36m/bin ...
py_versions_str=`cd ${DIR} && /opt/python/cp36-cp36m/bin/python -c "import build; print(build.get_centos_py_versions())"`
py_versions=( ${py_versions_str} )

# Delete previous builds (for testing locally)
rm -rf wheelhouse
rm -rf /io/wheelhouse/

echo "DEEPDRIVE_VERSION is ${DEEPDRIVE_VERSION}"

# Compile wheels
for PYBIN in  "${py_versions[@]}"; do
    "${PYBIN}/pip" install -r /io/DeepDrivePython/dev-requirements.txt

    # Call setup.py
    "${PYBIN}/pip" wheel /io/DeepDrivePython -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/deepdrive-*.whl; do
    auditwheel repair "$whl" -w /io/wheelhouse/
done

# Test installing packages
for PYBIN in  "${py_versions[@]}"; do
    "${PYBIN}/pip" install deepdrive --no-index -f /io/wheelhouse
done

# Upload to PyPi
for whl in /io/wheelhouse/deepdrive*manylinux1*.whl; do
    ${PYBIN}/twine upload -u ${PYPI_USERNAME} -p ${PYPI_PASSWORD} "$whl"
done
