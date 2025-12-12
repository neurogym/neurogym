import functools
import os

from neurogym.utils.logging import logger


def suppress_during_pytest(
    *exceptions: type[BaseException],
    message: str | None = None,
    reason: str | None = None,  # noqa: ARG001
):
    """Logs and suppresses specific exceptions during pytest, re-raises otherwise.

    Args:
        *exceptions: Exception types to suppress.
        message: Optional message to log when an exception occurs.
        reason: Optional reason for the suppression, not used in the decorator but can be useful for documentation.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                logger.error(f"Error {type(e)} while running '{func.__name__}'.")
                if not os.getenv("PYTEST_CURRENT_TEST"):  # re-raise if not in pytest
                    raise
                if message:
                    logger.error(message)

        return wrapper

    return decorator
