# Usage:
#
# docker run -it -v `pwd`/../..:/home/ue4/deepdrive-sim deepdriveio/deepdrive-ue4 python3 package.py
#
# To cache things:
# docker commit `docker ps --latest --format "{{.ID}}"` deepdriveio/deepdrive-ue4
#
# This uses a volume to avoid expensive context uploading and copies.
# Up to us to ensure sources are clean, wherever we're mounting them from.
# CI services like Travis or Jenkins will take care of this for us, but could be tricky on dev machines.
#
# To start from scratch
# docker build -t deepdriveio/deepdrive-ue4 .

FROM deepdriveio/ue4-full:4.21.1

ENV DEEPDRIVE_UNREAL_SOURCE_DIR /home/ue4/UnrealEngine

WORKDIR /home/ue4/deepdrive-sim/Packaging/docker

#RUN PKG_STAGE=requirements python3 package.py
#RUN PKG_STAGE=uepy_src     python3 package.py
#RUN PKG_STAGE=uepy_bin     python3 package.py
#RUN PKG_STAGE=package      python3 package.py
