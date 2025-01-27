# --------------------------------------
from datetime import datetime
from datetime import UTC

# --------------------------------------
from pathlib import Path


def mkdir(path: Path | str) -> Path:
    """
    A very thin shim over the Path class to
    create a directory if it doesn't exist and
    return the resolved and expanded path.

    Args:
        path (Path | str):
            A directory as a path or a string.

    Returns:
        Path:
            The resolved and expanded path.
    """

    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    return path.expanduser().resolve().absolute()


def timestamp(ms: bool = False) -> tuple[str, str]:
    """
    Create a datestamp and a timestamp as formatted strings.

    Args:
        ms (bool, optional):
            Use millisecond precision. Defaults to False.

    Returns:
        tuple[str, str]:
            A tuple containing:
                1. The formatted date.
                2. The formatted time.
    """

    # Simplified ISO format (no timezone, etc.).
    fmt = "%Y-%m-%d_%H-%M-%S"
    end = None

    if ms:
        # Use ms precision
        fmt += ":%f"
        end = -3

    # UTC time is used to avoid ambiguity.
    return datetime.strftime(datetime.now(UTC), fmt)[:end]
