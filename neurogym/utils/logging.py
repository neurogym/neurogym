from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from loguru import logger
from loguru._colorizer import AnsiParser

if TYPE_CHECKING:
    from loguru import Record


def _custom_format(record: Record) -> str:
    """Put together a context-aware custom format string.

    Args:
        record: The log record to process.

    Returns:
        A  Loguru formatting string.
    """
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
    format=_custom_format,
    level="INFO",
)
