TAG=deepdriveio/private:deepdrive-sim-package

RUN_ARGS=--net=host

build:
	docker build --network=host -t $(TAG) .

run:
	docker run $(RUN_ARGS) -it -e DEEPDRIVE_COMMIT=0a2ae17f0583f65b2dd76190dc3e6394a32c0c8f -e DEEPDRIVE_BRANCH=v3 -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} $(TAG)

bash:
	docker run $(RUN_ARGS) -it $(TAG) bash

push:
	docker push $(TAG)
