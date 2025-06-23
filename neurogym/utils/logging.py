import sys

from loguru import logger
from loguru._colorizer import AnsiParser


def custom_format(record):
    message = record["message"]

    # give custom color to log message
    color = record["extra"].get("color", "level")
    if color not in list(AnsiParser._foreground) + list(AnsiParser._background):  # noqa: SLF001
        color = "level"
    message = f"<{color}>{message}</{color}>"

    # give custom style (bold, etc) to log message
    style = record["extra"].get("style", "level")
    if style not in AnsiParser._style:  # noqa: SLF001
        style = "level"
    message = f"<{style}>{message}</{style}>"

    return f"<magenta>Neurogym</magenta> | <cyan>{record['time']:YYYY-MM-DD@HH:mm:ss}</cyan> | {message}\n"


logger.remove()
logger.add(
    sys.stderr,
    format=custom_format,
    level="INFO",
)
