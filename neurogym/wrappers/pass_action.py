from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from gymnasium import Wrapper

from neurogym.utils import spaces

if TYPE_CHECKING:
    from neurogym.core import TrialEnv


class PassAction(Wrapper):
    """Modifies observation by adding the previous action."""

    metadata: dict[str, str | None] = {  # noqa: RUF012
        "description": "Modifies observation by adding the previous action.",
        "paper_link": None,
        "paper_name": None,
    }

    def __init__(self, env: TrialEnv) -> None:
        super().__init__(env)
        self.env = env
        if isinstance(env.observation_space, spaces.Discrete):
            env_oss = int(env.observation_space.n)  # Number of discrete states
            self.observation_space = spaces.Discrete(n=env_oss + 1)
        elif env.observation_space.shape is not None:
            env_oss = env.observation_space.shape[0]
            self.observation_space = spaces.Box(
                -np.inf,
                np.inf,
                shape=(env_oss + 1,),
                dtype=np.float32,
            )
        else:
            msg = "env.observation_space.shape for PassAction may not be None"
            raise ValueError(msg)

    def reset(self, seed=None, options=None):
        step_fn = options.get("step_fn") if options else None
        if step_fn is None:
            step_fn = self.step
        return self.env.reset(options={"step_fn": step_fn}, seed=seed)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(obs, np.ndarray):
            obs = np.concatenate((obs, np.array([action])))
        else:
            obs = np.concatenate((np.array([obs]), np.array([action])))
        return obs, reward, terminated, truncated, info
