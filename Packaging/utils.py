import os
import tempfile
import zipfile


import logging as log
log.basicConfig(level=log.INFO)


import requests
from clint.textui import progress


def download(url, directory, warn_existing=True, overwrite=False):
    """Useful for downloading a folder / zip file from dropbox/s3/cloudfront and unzipping it to path"""
    if has_stuff(directory):
        if warn_existing:
            print('%s exists, do you want to re-download and overwrite any existing files (y/n)?' % directory, end=' ')
            overwrite = input()
            if 'n' in overwrite.lower():
                print('USING EXISTING %s - Try rerunning and overwriting if you run into problems.' % directory)
        elif not overwrite:
            print('Already exists, aborting download, pass overwrite=True to download over existing')
            return
    else:
        os.makedirs(directory, exist_ok=True)

    log.info('Downloading %s to %s...', url, directory)

    request = requests.get(url, stream=True)
    filename = url.split('/')[-1]
    if '?' in filename:
        filename = filename[:filename.index('?')]
    location = os.path.join(tempfile.gettempdir(), filename)
    with open(location, 'wb') as f:
        if request.status_code == 404:
            raise RuntimeError('Download URL not accessible %s' % url)
        total_length = int(request.headers.get('content-length'))
        for chunk in progress.bar(request.iter_content(chunk_size=1024), expected_size=(total_length / 1024) + 1):
            if chunk:
                f.write(chunk)
                f.flush()

    log.info('done.')
    zip_ref = zipfile.ZipFile(location, 'r')
    log.info('Unzipping temp file %s to %s...', location, directory)
    try:
        zip_ref.extractall(directory)
    except Exception:
        print('You may want to close all programs that may have these files open or delete existing '
              'folders this is trying to overwrite')
        raise
    finally:
        zip_ref.close()
        os.remove(location)
        log.info('Removed temp file %s', location)


def has_stuff(path):
    return os.path.exists(path) and (dir_has_stuff(path) or file_has_stuff(path))


def dir_has_stuff(path):
    return os.path.isdir(path) and os.listdir(path)


def file_has_stuff(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


def run_command_async(cmd, throw=True):
    """Allows streaming stdout and stderr to user while command executes"""
    from sarge import run, Capture
    # TODO: p = run(..., stdout=Capture(buffer_size=-1), stderr=Capture(buffer_size=-1))
    # TODO: Then log p.stdout. while process not complete in realtime and to file
    p = run(cmd, async_=True)
    p.close()
    if p.returncode != 0:
        if throw:
            raise RuntimeError('Command failed, see above')