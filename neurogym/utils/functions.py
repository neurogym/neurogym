from datetime import datetime, timezone
from pathlib import Path


def ensure_dir(path: Path | str) -> Path:
    """A thin shim over the Path class.

    Creates a directory if it doesn't exist and
    returns the resolved and expanded path
    to that directory.

    Args:
        path: A directory as a path or a string.

    Returns:
        The path to the directory.
    """
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    return path.expanduser().resolve().absolute()


def iso_timestamp() -> str:
    """Create a date-timestamp as a simplified ISO-formatted string.

    Useful for adding a unique but meaningful string to the
    name of a directory or a file that might be created
    repeatedly with the same name (for instance, when
    running the same experiment multiple times).

    NOTE: UTC time is used to avoid ambiguity.

    Returns:
        The formatted timestamp.
    """
    # Simplified ISO format (no timezone, etc.).
    return datetime.strftime(datetime.now(timezone.utc), "%Y-%m-%d_%H-%M-%S")
