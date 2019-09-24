TAG=deepdriveio/private:deepdrive-sim-package

build:
	docker build --network=host -t $(TAG) .

bash:
	docker run $(RUN_ARGS) -it $(TAG) bash

push:
	docker push $(TAG)
