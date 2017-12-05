from __future__ import print_function
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
    p = Popen(['git', 'log', '-1', '--format=%ci'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    out = ''.join(map(chr, out)).strip()
    if p.returncode != 0:
        raise RuntimeError('Could not get most recent commit')
    last_commit_time = git_commit_time_parse(out)
    last_commit_timestamp = last_commit_time.strftime('%Y%m%d%H%M%S')
    with open(os.path.join(DIR, 'VERSION')) as f:
        version = '%s.%s' % (f.read(), last_commit_timestamp)
    return version

if __name__ == '__main__':
    print(get_package_version())
