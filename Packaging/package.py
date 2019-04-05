#!/usr/bin/env python3
from ue4helpers import FilesystemUtils, ProjectPackager, UnrealUtils
from os.path import abspath, dirname, join
import shutil


def main():
    # Compute the absolute path to the root of the repository
    root = dirname(dirname(abspath(__file__)))

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

    # Compress the packaged distribution
    archive = packager.archive()

    # Rename archive to format used on S3
    archive = shutil.move(archive, reformat_name(archive))

    # TODO: upload the archive to Amazon S3
    print('Created compressed archive "{}".'.format(archive))


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
    name, platform, version_and_ext = archive_name.split('-')
    name = name.lower() + '-sim'
    platform = platform.lower()
    ret = '-'.join([name, platform, version_and_ext])
    return ret


if __name__ == '__main__':
    main()

