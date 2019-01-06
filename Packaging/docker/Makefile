.PHONY: package install commit clean

install: package commit

clean:
	docker build -t deepdriveio/deepdrive-ue4 .

commit:
	docker commit `docker ps --latest --format "{{.ID}}"` deepdriveio/deepdrive-ue4

package:
	-docker run -it -v `pwd`/../..:/home/ue4/deepdrive-sim deepdriveio/deepdrive-ue4 python3 package.py
