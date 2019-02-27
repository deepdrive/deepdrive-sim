import os
import sys
import pip

from pathlib import Path
import importlib
import traceback

REQ_DIR_NAME = 'UEPyPackages'
REQUIREMENTS = [
    dict(module='zmq', pip='pyzmq'),
    dict(module='pyarrow', pip='pyarrow'),
    dict(module='requests', pip='requests'),
]


def ensure_requirements():
    curr_dir = os.path.dirname(get_this_filename())
    req_dir = str(Path(curr_dir).parent.parent.joinpath(REQ_DIR_NAME))
    if req_dir not in sys.path:
        sys.path.insert(0, req_dir)
    for req in REQUIREMENTS:
        module = req['module']
        pip_name = req['pip']
        try:
            importlib.import_module(module)
        except ImportError:
            print('Could not import %s. Installing to %s...' % (module, req_dir))
            pip_install(pip_name, req_dir)
        else:
            print('Found %s' % pip_name)


def pip_install(package, dirname):
    if hasattr(pip, 'main'):
        pip_main = pip.main
    else:
        from pip._internal import main as pip_main

    try:
        pip_main(['install', '-vvv', package])
    except Exception as e:
        # Swallow exceptions here to ignore pip-req-tracker tmp deletion errors.
        # Attempting to import said module will hopefully keep this from being hard to track root cause for.
        # TODO: Be more specific about exceptions we swallow.
        print('Error installing %s - error was: %s' % (package, str(e)))


def get_this_filename():
    # Hack to get current file path in UnrealEditor
    try:
        raise NotImplementedError("No error")
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        filename = traceback.extract_tb(exc_traceback)[-1].filename
    return filename


if __name__ == '__main__':
    # import os
    # print(os.system('/media/a/data-ext4/UnrealEngine/Engine/Plugins/Marketplace/UnrealEnginePython/EmbeddedPython/Linux/bin/python3 -m pip install sarge'))
    # TODO: Try to create a tempfile within Unreal using the Pip TempFile class

    import tempfile
    import shutil

    dirpath = tempfile.mkdtemp()
    print('tmp file ' + str(dirpath))
    pass
    # ensure_requirements()
