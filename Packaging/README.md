# Packaging

Package the project in Unreal on Linux to ~/deepdrive-packaged, or on Windows to %HOMEPATH%\deepdrive-packaged
choosing 32 bit for Windows as we still fit in 2GB of program memory.

Copy `package.sh` into the LinuxNoEditor directory and run it 
TODO: `package.bat`

Then copy `run.sh` into the `LinuxNoEditor` directory and zip up _that_ directory for upload to
s3 and CloudFront.

TODO: `WindowsNoEditor`