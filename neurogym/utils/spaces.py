from gymnasium.spaces import Box as GymBox
from gymnasium.spaces import Dict, MultiBinary, MultiDiscrete, Space, Tuple
from gymnasium.spaces import Discrete as GymDiscrete
from gymnasium.spaces.utils import flatdim, flatten, unflatten


class Box(GymBox):
    """Thin wrapper of gymnasium.spaces.Box.

    Allow the user to give names to each dimension of the Box.

    Args:
        low, high, kwargs: see gymnasium.spaces.Box
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
    """Thin wrapper of gymnasium.spaces.Discrete.

    Allow the user to give names to each dimension of the Discrete space.

    Args:
        low, high, kwargs: see gymnasium.spaces.Box
        name: dict describing the name of different dimensions

    Example usage:
        observation_space = Discrete(n=3, name={'fixation': 0, 'stimulus': [1, 2]})
    """

    def __init__(self, n, name=None, **kwargs):
        super().__init__(n)
        if name is not None:
            assert isinstance(name, dict)
            self.name = name


__all__ = [
    "Space",
    "Box",
    "Discrete",
    "MultiDiscrete",
    "MultiBinary",
    "Tuple",
    "Dict",
    "flatdim",
    "flatten",
    "unflatten",
]
