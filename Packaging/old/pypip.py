import os
import sys
import pip

from pathlib import Path
import importlib
import traceback


# TODO: versions
# TODO: Check for local modules that shadow PyPI package names
# TODO: Don't use sys.path for target - do something more persistent

def ensure(requirements, req_path=None):
    if req_path is not None and req_path not in sys.path:
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
        except (ModuleNotFoundError, ImportError):
            print('Could not import %s' % module)
            if req_path is not None:
                print('Installing to %s' % req_path)
            pip_install(pip_name, req_path)
        else:
            print('Found %s' % pip_name)

    if req_path is not None:
        # TODO: Use site.addsitedir, PYTHONPATH, .pth files, or site.PREFIXES as below does not work
        raise NotImplementedError('Custom requirement directory not supported')
    else:
        # Reload packages
        import site
        importlib.reload(site)


def pip_install(package, req_path=None):
    if hasattr(pip, 'main'):
        pip_main = pip.main
    else:
        from pip._internal import main as pip_main

    args = ['install']
    if req_path is not None:
        args += ['--target', req_path]
        # Ubuntu bug workaround - https://github.com/pypa/pip/issues/3826#issuecomment-427622702
        args.append('--system')

    args.append(package)
    pip_main(args)


def get_this_filename():
    # Hack to get current file path in UnrealEditor
    try:
        raise NotImplementedError("No error")
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        filename = traceback.extract_tb(exc_traceback)[-1].filename
    return filename
