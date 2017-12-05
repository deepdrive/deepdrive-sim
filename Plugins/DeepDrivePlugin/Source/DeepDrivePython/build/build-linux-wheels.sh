#!/usr/bin/env bash
set -euo pipefail

# Compile with Python 3.5+
# TODO: Update dynamically for > 3.6
py_versions=( "/opt/python/cp35-cp35m/bin" "/opt/python/cp36-cp36m/bin" )

# Delete previous builds (for testing locally)
rm -rf wheelhouse
rm -rf /io/wheelhouse/

echo "DEEPDRIVE_VERSION is ${DEEPDRIVE_VERSION}"

# Compile wheels
for PYBIN in  "${py_versions[@]}"; do
    "${PYBIN}/pip" install -r /io/DeepDrivePython/dev-requirements.txt
    "${PYBIN}/pip" wheel /io/DeepDrivePython -w wheelhouse/  # Calls setup.py
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    auditwheel repair "$whl" -w /io/wheelhouse/
done

# Test installing packages
for PYBIN in  "${py_versions[@]}"; do
    "${PYBIN}/pip" install deepdrive --no-index -f /io/wheelhouse
done

# Upload to PyPi
if [ "${DEEPDRIVE_BRANCH}" == "release" ]; then
    for whl in /io/wheelhouse/deepdrive*manylinux1*.whl; do
        ${PYBIN}/twine upload -u ${PYPI_USERNAME} -p ${PYPI_PASSWORD} "$whl"
    done
else
    echo Not on release branch, so not pushing to PyPi
fi