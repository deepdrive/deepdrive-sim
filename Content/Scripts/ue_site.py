"""
On Editor/Engine start, the ue_site module is tried for import. You should place initialization code there.
If the module cannot be imported, you will get a (harmful) message in the logs.

This file is also necessary for packaged builds
"""

# TODO: Perhaps download python libs here with:
# import os
# # noinspection PyCompatibility
# import urllib.request
# from io import BytesIO
# from zipfile import ZipFile
# from pathlib import Path
#
# CURR_PATH = Path(__file__).resolve().parent
# LIB_PATH = CURR_PATH.parent.parent / 'python_libs'
# LIB_URL = 'https://s3-us-west-1.amazonaws.com/deepdrive/unreal_python_lib/python_libs.zip'
#
#
# def main():
#     if not LIB_PATH.exists() or next(LIB_PATH.iterdir(), None) is None:
#         LIB_PATH.mkdir(exist_ok=True)
#         print('Downloading Python libs (71MB) for Unreal embedded Python from', LIB_URL, '...')
#         download_zip(LIB_URL, str(LIB_PATH))
#
#
# def download_zip(url, path):
#     """Download a zip file and unzipping it to path"""
#     with urllib.request.urlopen(url) as response:
#         content = response.read()
#         zip_ref = ZipFile(BytesIO(content))
#         try:
#             zip_ref.extractall(path)
#         except Exception:
#             print('You may want to close all programs that may have these files open or delete existing '
#                   'folders this is trying to overwrite')
#             raise
#         finally:
#             zip_ref.close()
#
#
# main()