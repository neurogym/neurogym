import numpy as np
from gymnasium import spaces

import neurogym as ngym


class Null(ngym.TrialEnv):
    """Null task."""

    def __init__(self, dt=100) -> None:
        super().__init__(dt=dt)
        self.action_space = spaces.Discrete(1)
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(1,),
            dtype=np.float32,
        )

    def _new_trial(self, **kwargs):
        trial = {}
        trial.update(kwargs)
        return trial

    def _step(self, action):  # noqa: ARG002
        return 0, 0, False, False, {}
