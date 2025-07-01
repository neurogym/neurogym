import functools
import os

from neurogym.utils.logging import logger


def suppress_during_pytest(*exceptions: type[BaseException], message: str | None = None):
    """Logs and suppresses specific exceptions during pytest, re-raises otherwise.

    Args:
        *exceptions: Exception types to suppress.
        message: Optional message to log when an exception occurs.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions:
                logger.error(f"Error while running '{func.__name__}'.")
                if message:
                    logger.error(message)
                if not os.getenv("PYTEST_CURRENT_TEST"):  # re-raise if not in pytest
                    raise

        return wrapper

    return decorator
