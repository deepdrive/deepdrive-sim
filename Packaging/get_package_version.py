from __future__ import print_function

import platform

from subprocess import Popen, PIPE
import os
from datetime import datetime, timedelta

DIR = os.path.dirname(os.path.realpath(__file__))


def git_commit_time_parse(t):
    ret = datetime.strptime(t[0:19], '%Y-%m-%d %H:%M:%S')
    if t[20] == '+':
        ret -= timedelta(hours=int(t[21:23]), minutes=int(t[23:25]))
    elif t[20] == '-':
        ret += timedelta(hours=int(t[21:23]), minutes=int(t[23:25]))
    return ret


def get_package_version():
    version_path = os.path.join(os.path.dirname(DIR), 'Content', 'Data')

    git_dir = os.path.join(os.path.dirname(DIR), '.git')
    if not os.path.exists(git_dir):
        # We are not in a git repo. Just read the last version - nothing is going to be pushed here.
        with open(os.path.join(version_path, 'VERSION')) as fr:
            version = fr.read()
    else:
        # The %ci option gives the committer date which guarantees HEAD is most recent
        p = Popen(['git', '--git-dir', git_dir, 'log', '-1', '--format=%ci'],
                  stdin=PIPE, stdout=PIPE, stderr=PIPE)
        out, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError('Could not get most recent commit')

        if not isinstance(out, str):
            out = ''.join(map(chr, out))
        out = out.strip()

        last_commit_time = git_commit_time_parse(out)
        last_commit_timestamp = last_commit_time.strftime('%Y%m%d%H%M%S')

        with open(os.path.join(version_path, 'MAJOR_MINOR_VERSION')) as fr:
            version = '%s.%s' % (fr.read().strip(), last_commit_timestamp)
        with open(os.path.join(version_path, 'VERSION'), 'w') as fw:
            fw.write(version)
    return version


def get_package_filename():
    return 'deepdrive-sim-%s-%s.zip' % (platform.system().lower(),
                                        get_package_version())


if __name__ == '__main__':
    print(get_package_version())
