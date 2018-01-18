from __future__ import print_function

import os
import shutil
import sys
import tempfile

import boto
import boto.s3
from boto.s3.key import Key
from get_package_version import get_package_version

DIR = os.path.dirname(os.path.realpath(__file__))


def main():
    out_filename = 'deepdrive-sim-windows-%s' % get_package_version()
    out_path = os.path.join(tempfile.gettempdir(), out_filename)
    in_path = os.path.join(os.path.expanduser('~'), 'Deepdrive', 'sim')
    print('Zipping %s to %s.zip...' % (in_path, out_path))
    shutil.make_archive(out_path, 'zip', in_path)
    out_path += '.zip'
    out_filename += '.zip'
    upload_s3(out_path, out_filename)
    print('Upload successful, deleting temp zip file %s' % out_path)
    os.remove(out_path)


def upload_s3(path, key):
    bucket_name = 'deepdrive'
    conn = boto.connect_s3()
    bucket = conn.get_bucket(bucket_name)

    print('Uploading %s to Amazon S3 bucket %s' % (path, bucket_name))

    def percent_cb(complete, total):
        sys.stdout.write('%r of %r\n' % (complete, total))
        sys.stdout.flush()

    k = Key(bucket)
    k.key = 'sim/' + key
    k.set_contents_from_filename(path, cb=percent_cb, num_cb=10)


if __name__ == '__main__':
    main()
