# ruff: noqa: E402

from importlib.util import find_spec

# check if optional dependencies are installed
_SB3_INSTALLED: bool = find_spec("stable_baselines3") is not None and find_spec("sb3_contrib") is not None
_PSYCHOPY_INSTALLED: bool = find_spec("psychopy") is not None

from .envs.registration import make, register
from .utils import info, spaces
from .utils.data import Dataset

__version__ = "2.2.0"
