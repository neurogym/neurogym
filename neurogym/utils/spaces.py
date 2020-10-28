from gym.spaces.space import Space
from gym.spaces.box import Box as GymBox
from gym.spaces.discrete import Discrete as GymDiscrete
from gym.spaces.multi_discrete import MultiDiscrete
from gym.spaces.multi_binary import MultiBinary
from gym.spaces.tuple import Tuple
from gym.spaces.dict import Dict

from gym.spaces.utils import flatdim
from gym.spaces.utils import flatten
from gym.spaces.utils import unflatten


class Box(GymBox):
    """Thin wrapper of gym.spaces.Box.

    Allow the user to give names to each dimension of the Box.

    Args:
        low, high, kwargs: see gym.spaces.Box
        name: dict describing the name of different dimensions

    Example usage:
        observation_space = Box(low=0, high=1,
                                name={'fixation': 0, 'stimulus': [1, 2]})
    """
    def __init__(self, low, high, name=None, **kwargs):
        super().__init__(low, high, **kwargs)
        if name is not None:
            assert isinstance(name, dict)
            self.name = name


class Discrete(GymDiscrete):
    """Thin wrapper of gym.spaces.Discrete.

    Allow the user to give names to each dimension of the Discrete space.

    Args:
        low, high, kwargs: see gym.spaces.Box
        name: dict describing the name of different dimensions

    Example usage:
        observation_space = Discrete(n=3, name={'fixation': 0, 'stimulus': [1, 2]})
    """
    def __init__(self, n, name=None, **kwargs):
        super().__init__(n)
        if name is not None:
            assert isinstance(name, dict)
            self.name = name


__all__ = ["Space", "Box", "Discrete", "MultiDiscrete", "MultiBinary", "Tuple", "Dict", "flatdim", "flatten", "unflatten"]
