from datetime import UTC, datetime
from pathlib import Path


def mkdir(path: Path | str) -> Path:
    """A thin shim over the Path class.

    Creates a directory if it doesn't exist and
    returns the resolved and expanded path.

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


def timestamp(ms: bool = False) -> str:
    """Create a datestamp and a timestamp as formatted strings.

    Args:
        ms (bool, optional):
            Use millisecond precision. Defaults to False.

    Returns:
        str:
            The formatted timestamp.
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
