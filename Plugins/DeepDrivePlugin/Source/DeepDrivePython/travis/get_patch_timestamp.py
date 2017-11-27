import sys
from datetime import datetime

import pytz

print(datetime.strptime(sys.argv[1], '%Y-%m-%d %H:%M:%S %z').
      astimezone(pytz.utc).strftime('%Y%m%d%H%M%S'))
