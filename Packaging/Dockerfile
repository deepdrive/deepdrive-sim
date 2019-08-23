# This is meant to run in CI and independent of the host filesystem

# docker build -t deepdriveio/deepdrive-sim-build .
# docker run -e DEEPDRIVE_BRANCH=v3_stable -e DEEPDRIVE_COMMIT=4c198c5a17b7e3b24964d9b1369372647e5e82a3 -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY} -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} deepdriveio/deepdrive-sim-build

FROM deepdriveio/ue4-deepdrive-deps:latest
RUN pip3 install ue4-ci-helpers awscli
CMD curl https://raw.githubusercontent.com/deepdrive/deepdrive-sim/${DEEPDRIVE_COMMIT}/Packaging/ci-package.sh | sudo bash
