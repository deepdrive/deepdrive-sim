# This is meant to run in CI and independent of the host filesystem

# docker run -it --net=host -e DEEPDRIVE_BRANCH=v3 -e DEEPDRIVE_COMMIT=bbaf0b48a233d4dc341f3382932f45130b760011 -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY} -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} deepdriveio/private:deepdrive-sim-package
# docker push deepdriveio/private:deepdrive-sim-package

# TODO: Build this every time CI is run?

FROM deepdriveio/ue4-deepdrive-deps:latest

WORKDIR /home/ue4
RUN git clone https://github.com/deepdrive/deepdrive-sim

COPY requirements.txt ./requirements-lock.txt
RUN pip3 install -r requirements-lock.txt


# Bootstrap source into build container
CMD curl -s -N https://raw.githubusercontent.com/deepdrive/deepdrive-sim/${DEEPDRIVE_COMMIT}/Packaging/ci-package.sh | bash
