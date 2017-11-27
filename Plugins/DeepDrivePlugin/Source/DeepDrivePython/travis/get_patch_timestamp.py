import sys
from datetime import datetime, timedelta


def git_commit_time_parse(t):
    ret = datetime.strptime(t[0:19], '%Y-%m-%d %H:%M:%S')
    if t[20] == '+':
        ret -= timedelta(hours=int(t[21:23]), minutes=int(t[23:25]))
    elif t[20] == '-':
        ret += timedelta(hours=int(t[21:23]), minutes=int(t[23:25]))
    return ret


print(git_commit_time_parse(sys.argv[1]).strftime('%Y%m%d%H%M%S'))
