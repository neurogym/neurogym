try:
    import sb3_contrib
    import stable_baselines3

    _SB3_INSTALLED = True
except ImportError:
    _SB3_INSTALLED = False


from .envs.registration import make, register
from .utils import info, spaces
from .utils.data import Dataset

__version__ = "2.1.0"
