TAG=deepdriveio/private:deepdrive-sim-package

build:
	docker build --network=host -t $(TAG) .

package:
	docker run -it --net=host -e DEEPDRIVE_BRANCH=v3_stable -e DEEPDRIVE_COMMIT=${DEEPDRIVE_COMMIT} -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY} -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} $(TAG)

push:
	docker push $(TAG)
