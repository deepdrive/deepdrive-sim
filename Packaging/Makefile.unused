TAG=deepdriveio/private:deepdrive-sim-package

build:
	docker build --network=host -t $(TAG) .

package:
	docker run -it --net=host $(TAG)

push:
	docker push $(TAG)
