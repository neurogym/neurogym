import sys

from loguru import logger

logger.remove()
logger.add(
    sys.stderr,
    format="<magenta>Neurogym</magenta> | <cyan>{time:YYYY-MM-DD@HH:mm:ss}</cyan> | <level>{message}</level>",
    level="INFO",
)
