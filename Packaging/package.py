#!/usr/bin/env python3
import os
import stat
import sys

import boto
import boto.s3
from boto.s3.key import Key
from ue4helpers import FilesystemUtils, ProjectPackager, UnrealUtils
from os.path import abspath, dirname, join
import shutil


def main():
    # Compute the absolute path to the root of the repository
    root = dirname(dirname(abspath(__file__)))

    # This is already done via using the ue4-deepdrive-deps base image
    # install_plugins(root)

    # Create our project packager
    packager = ProjectPackager(
        root=root,
        version=FilesystemUtils.read(join(root, 'Content', 'Data', 'VERSION')),
        archive='{name}-{platform}-{version}',
    )
    # Clean any previous build artifacts
    packager.clean()

    # Package the project
    packager.package(args=['Development'])

    # Install dependencies like zmq and arrow of UEPy
    install_uepy_requirements(join(root, 'dist'))

    # Compress the packaged distribution
    archive = packager.archive()

    # Rename archive to format used on S3
    archive = shutil.move(archive, reformat_name(archive))

    print('Created compressed archive "{}".'.format(archive))

    if '--upload' in sys.argv:
        upload_s3(archive, archive.split('/')[-1])


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


def upload_s3(filepath, filename):
    bucket_name = 'deepdrive'
    conn = boto.connect_s3()
    bucket = conn.get_bucket(bucket_name)

    print('Uploading %s to Amazon S3 bucket %s' % (filepath, bucket_name))

    def percent_cb(complete, total):
        sys.stdout.write('%r of %r\n' % (complete, total))
        sys.stdout.flush()

    k = Key(bucket)
    k.key = 'sim/' + filename
    k.set_contents_from_filename(filepath, cb=percent_cb, num_cb=10)

    print(f'Finished upload to '
          f'https://s3-us-west-1.amazonaws.com/deepdrive/sim/{filename}')


if __name__ == '__main__':
    main()

