import sys
from os.path import dirname, realpath, join

DIR = dirname(realpath(__file__))
PACKAGE_DIR = join(dirname(DIR), 'Packaging')
sys.path.insert(0, PACKAGE_DIR)

from package import upload_s3_str, get_package_version

def main():
    sim_version = get_package_version()
    upload_s3_str(dest_filename=f'unvalidated-bindings-versions/{sim_version}',
                  content='')

if __name__ == '__main__':
    main()
