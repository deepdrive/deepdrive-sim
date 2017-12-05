from __future__ import print_function
import os
import shutil
import tempfile
import boto
import boto.s3
import sys
from boto.s3.key import Key

from get_package_version import get_package_version

DIR = os.path.dirname(os.path.realpath(__file__))


def main():
    version = get_package_version()
    output_filename = 'deepdrive-sim-windows-%s.zip' % version
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, output_filename)
    print('zipping %s to %s' % (output_path, temp_dir))
    shutil.make_archive(output_filename, 'zip', temp_dir)

def upload_s3(filename):
    bucket_name = 'deepdrive'
    conn = boto.connect_s3()
    bucket = conn.create_bucket(bucket_name)

    print('Uploading %s to Amazon S3 bucket %s' % (filename, bucket_name))

    def percent_cb(complete, total):
        sys.stdout.write('%r of %r\n' % (complete, total))
        sys.stdout.flush()

    k = Key(bucket)
    k.key = filename
    k.set_contents_from_filename(filename, cb=percent_cb, num_cb=10)


if __name__ == '__main__':
    main()
