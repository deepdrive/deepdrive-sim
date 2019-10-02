TAG=deepdriveio/private:deepdrive-sim-package

RUN_ARGS=--net=host

build:
	docker build --network=host -t $(TAG) .

run:
	docker run $(RUN_ARGS) -it -e DEEPDRIVE_COMMIT=df5d67062dbc8b25cdb275cdba8be44c730e9f62 -e DEEPDRIVE_BRANCH=v3 $(TAG)

bash:
	docker run $(RUN_ARGS) -it $(TAG) bash

push:
	docker push $(TAG)
