# Usage:
#
# docker run -t deepdriveio/deepdrive-ue4 -v deepdrive-sim:`pwd`/.. package.py
#
# Uses a volume to avoid expensive context uploading and copies.
# Up to us to ensure sources are clean, wherever we're mounting them from.
# CI services like Travis or Jenkins will take care of this for us, but could be tricky on dev machines.

FROM deepdriveio/ue4-full:4.21.1
COPY *.py /home/ue4/deepdrive-package/
#RUN python3 package.py

