import os
import sys
import pip

from pathlib import Path
import importlib
import traceback


# TODO: versions
# TODO: Check for local modules that shadow PyPI package names
# TODO: Don't use sys.path for target - do something more persistent

def ensure(requirements, requirements_dir=None):
    if requirements_dir is not None:
        curr_dir = os.path.dirname(get_this_filename())
        req_path = str(Path(curr_dir).parent.parent.joinpath(requirements_dir))
        if req_path not in sys.path:
            sys.path.insert(0, req_path)
    else:
        req_path = None
    for req in requirements:
        if isinstance(req, str):
            req = dict(module=req, pip=req)
        module = req['module']
        pip_name = req['pip']
        try:
            importlib.import_module(module)
        except ImportError:
            print('Could not import %s' % module)
            if req_path is not None:
                print('Installing to %s' % req_path)
            pip_install(pip_name, req_path)
        else:
            print('Found %s' % pip_name)


def pip_install(package, req_path=None):
    if hasattr(pip, 'main'):
        args = ['install']
        if req_path is not None:
            args += ['--target', req_path]
        pip.main(args.append(package))


def get_this_filename():
    # Hack to get current file path in UnrealEditor
    try:
        raise NotImplementedError("No error")
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        filename = traceback.extract_tb(exc_traceback)[-1].filename
    return filename
