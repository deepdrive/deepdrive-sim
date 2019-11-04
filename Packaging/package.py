#!/usr/bin/env python3
import json
import os
import stat
import sys
from glob import glob

import boto
import boto.s3
from boto.s3.key import Key
from retry import retry
from ue4helpers import FilesystemUtils, ProjectPackager, UnrealUtils
from os.path import abspath, dirname, join
import shutil
from google.cloud import storage

from get_package_version import get_package_filename, \
    get_package_version

ROOT = dirname(dirname(abspath(__file__)))
VERSION = get_package_version()
AWS_DEEPDRIVE_BUCKET_NAME = 'deepdrive'
GCS_DEEPDRIVE_BUCKET_NAME = 'deepdriveio'
RELEASE_PATH = 'sim'
RELEASE_CANDIDATE_PATH = 'sim/release_candidates'
DEEPDRIVE_PACKAGE_NO_UPLOAD = 'DEEPDRIVE_PACKAGE_NO_UPLOAD' in os.environ

def main():

    # This is already done via using the ue4-deepdrive-deps base image
    # install_plugins(root)

    # Create our project packager
    packager = ProjectPackager(
        root=ROOT,
        version=VERSION,
        archive='{name}-{platform}-{version}',
    )
    # Clean any previous build artifacts
    packager.clean()

    # Package the project
    packager.package(args=['Development'])

    # Install dependencies like zmq and arrow of UEPy
    install_uepy_requirements(join(ROOT, 'dist'))

    # Compress the packaged distribution
    archive = packager.archive()

    # Rename archive to format used on S3
    archive = shutil.move(archive, reformat_name(archive))

    print('Created compressed archive "{}".'.format(archive))

    if '--upload' in sys.argv and not DEEPDRIVE_PACKAGE_NO_UPLOAD:
        s3_url, gcs_url = upload_s3_and_gcs(archive, archive.split('/')[-1])
        if 'IS_DEEPDRIVE_SIM_BUILD' in os.environ:
            json_out = json.dumps(dict(s3_url=s3_url, gcs_url=gcs_url))
            print(f'|~__JSON_OUT_LINE_DELIMITER__~| {json_out}')


def install_uepy_requirements(dist):
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    uepy = ensure_uepy_executable(dist)
    command = '{uepy} -m pip install -r {curr_dir}/uepy-requirements.txt'. \
        format(uepy=uepy, curr_dir=curr_dir)
    print('Installing UEPY dependencies: %s ' % command)
    result = os.system(command)
    if result != 0:
        raise Exception('Could not install UEPY python requirements')


def install_plugins(root):
    # TODO: download our plugin zip files from AWS if the local versions don't exist

    # Download and extract the prebuilt binaries for the Substance plugin
    print('Downloading and extracting the prebuilt Substance plugin...')
    UnrealUtils.install_plugin(join(root, 'Packaging', 'Substance-4.21.0.31-Desktop.zip'), 'Substance',
                               prefix='Marketplace')

    # Download and extract the prebuilt binaries for  the UnrealEnginePython plugin
    print('Downloading and extracting the prebuilt UnrealEnginePython plugin...')
    UnrealUtils.install_plugin(join(root, 'Packaging', 'UnrealEnginePython-20190128-Linux.zip'), 'UnrealEnginePython')


def reformat_name(archive_name):
    name, platform, version_and_ext = archive_name.split('/')[-1].split('-')
    name = name.lower() + '-sim'
    platform = platform.lower()
    ret = '-'.join([name, platform, version_and_ext])
    return ret


def get_uepy_path(sim_path):
    ret = os.path.join(
        sim_path,
        'LinuxNoEditor/Engine/Plugins/UnrealEnginePython/EmbeddedPython/Linux/bin/python3')
    return ret


def ensure_uepy_executable(sim_path):
    """
    Ensure the UEPY python binary is executable
    :param path: Path to UEPy python binary
    :return:
    """
    uepy = get_uepy_path(sim_path)
    st = os.stat(uepy)
    os.chmod(uepy, st.st_mode | stat.S_IEXEC)
    return uepy


def upload_s3_and_gcs(source_path: str, dest_filename: str):
    s3_url = upload_s3(source_path, dest_filename)
    gcs_url = upload_gcs(source_path, dest_filename)
    return s3_url, gcs_url

@retry(tries=5)
def upload_gcs(source_path: str, dest_filename: str) -> str:
    print('Uploading %s to GCS bucket %s' % (source_path,
                                             GCS_DEEPDRIVE_BUCKET_NAME))
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(GCS_DEEPDRIVE_BUCKET_NAME)
    blob_name = f'{RELEASE_CANDIDATE_PATH}/{dest_filename}'
    bucket.blob(blob_name).upload_from_filename(source_path)

    url = f'https://storage.googleapis.com/deepdriveio/{blob_name}'
    print(f'Finished upload to {url}')
    return url


@retry(tries=5)
def upload_s3(source_path: str, dest_filename: str) -> str:
    conn = boto.connect_s3()
    bucket = conn.get_bucket(AWS_DEEPDRIVE_BUCKET_NAME)

    print('Uploading %s to Amazon S3 bucket %s' % (source_path,
                                                   AWS_DEEPDRIVE_BUCKET_NAME))

    def percent_cb(complete, total):
        sys.stdout.write('%r of %r\n' % (complete, total))
        sys.stdout.flush()

    k = Key(bucket)
    k.key = f'{RELEASE_CANDIDATE_PATH}/{dest_filename}'
    k.set_contents_from_filename(source_path, cb=percent_cb, num_cb=10)

    url = f'https://s3-us-west-1.amazonaws.com/deepdrive/{k.key}'
    print(f'Finished upload to {url}')

    return url


def upload_s3_str(content: str, dest_filename: str) -> str:
    conn = boto.connect_s3()
    bucket = conn.get_bucket(AWS_DEEPDRIVE_BUCKET_NAME)

    print(f'Uploading to Amazon S3 bucket {AWS_DEEPDRIVE_BUCKET_NAME}')

    def percent_cb(complete, total):
        sys.stdout.write('%r of %r\n' % (complete, total))
        sys.stdout.flush()

    k = Key(bucket)
    k.key = dest_filename
    k.set_contents_from_string(content, cb=percent_cb, num_cb=10)

    url = f'https://s3-us-west-1.amazonaws.com/deepdrive/{k.key}'
    print(f'Finished upload to {url}')

    return url


def copy_release_candidate_to_release():
    filename = get_package_filename()
    conn = boto.connect_s3()
    bucket = conn.get_bucket(AWS_DEEPDRIVE_BUCKET_NAME)

    src_key = Key(bucket)
    src_key.key = f'{RELEASE_CANDIDATE_PATH}/{filename}'
    dst_key = f'{RELEASE_PATH}/{filename}'
    print(f'Copying sim to release on s3 in "{AWS_DEEPDRIVE_BUCKET_NAME}" bucket...'
          f'\n\t{src_key.key} => {dst_key}')
    src_key.copy(AWS_DEEPDRIVE_BUCKET_NAME, dst_key, preserve_acl=True,
                 validate_dst_bucket=False)
    print('Copy complete')

def upload_latest_sim():
    # For local use
    filepaths = glob(f'{ROOT}/deepdrive-sim-linux-*')
    latest_filepath = max(filepaths, key=os.path.getctime)
    filename = str(latest_filepath.split('/')[-1])
    upload_s3_and_gcs(source_path=latest_filepath, dest_filename=filename)


if __name__ == '__main__':
    if '--upload-only' in sys.argv:
        upload_latest_sim()
    elif '--copy-release-candidate-to-release' in sys.argv:
        copy_release_candidate_to_release()
    else:
        main()

