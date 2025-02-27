from datetime import datetime, timezone
from pathlib import Path


def ensure_dir(path: Path | str) -> Path:
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


def iso_datetime(ms: bool = False) -> str:
    """Create a date-timestamp as an ISO-formatted string.

    Useful for adding a unique but meaningful string to the
    name of a directory or a file that might be created
    repeatedly with the same name (for instance, when
    running the same experiment multiple times).

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
    return datetime.strftime(datetime.now(timezone.utc), fmt)[:end]
