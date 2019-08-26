# This is meant to run in CI and independent of the host filesystem

# docker build --network=host -t deepdriveio/private:deepdrive-sim-package .
# docker run -it --net=host -e DEEPDRIVE_BRANCH=v3_stable -e DEEPDRIVE_COMMIT=bbaf0b48a233d4dc341f3382932f45130b760011 -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY} -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} deepdriveio/private:deepdrive-sim-package
# docker push deepdriveio/private:deepdrive-sim-package

FROM deepdriveio/ue4-deepdrive-deps:latest
RUN pip3 install ue4-ci-helpers awscli boto
WORKDIR /home/ue4

CMD curl -s -N https://raw.githubusercontent.com/deepdrive/deepdrive-sim/${DEEPDRIVE_COMMIT}/Packaging/ci-package.sh | bash
