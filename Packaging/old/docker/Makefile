.PHONY: package install commit clean bash

install: package commit

commit:
	docker commit `docker ps --latest --format "{{.ID}}"` deepdriveio/deepdrive-ue4

package:
	-docker run -e PKG_STAGE=all -v `pwd`/../..:/home/ue4/deepdrive-sim deepdriveio/deepdrive-ue4 python3 package.py

bash:
	docker run -it -v `pwd`/../..:/home/ue4/deepdrive-sim deepdriveio/deepdrive-ue4 bash

clean:
	docker build -t deepdriveio/deepdrive-ue4 .