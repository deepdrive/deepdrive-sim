from __future__ import print_function
from subprocess import Popen, PIPE
import os
from datetime import datetime, timedelta

DIR = os.path.dirname(os.path.realpath(__file__))

def get_package_version():
    version_path = os.path.join(os.path.dirname(DIR), 'Content', 'Data', 'VERSION')
    with open(version_path, 'r') as f:
        version = f.read()
    return version

if __name__ == '__main__':
    print(get_package_version())
