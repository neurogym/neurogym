from datetime import datetime, timezone
from pathlib import Path
from IPython import get_ipython

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
    return datetime.strftime(datetime.now(timezone.utc), fmt)[:end]


def is_notebook() -> bool:
    """Determine if the caller is running in a Jupyter notebook.

    Courtesy of https://stackoverflow.com/a/39662359/4639195.

    Returns:
        bool:
            True if running in a notebook.
    """
    try:
        shell = get_ipython().__class__.__name__
        match (shell):
            case "ZMQInteractiveShell":
                # Jupyter notebook or qtconsole
                return True
            case "TerminalInteractiveShell":
                # Terminal running IPython
                return False
            case _:
                # Other type (?)
                return False
    except NameError:
        # Probably standard Python interpreter
        return False
