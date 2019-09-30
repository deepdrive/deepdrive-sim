from loguru import logger as log
from botleague_helpers.logs import add_slack_error_sink

add_slack_error_sink(log, '#deepdrive-alerts')
