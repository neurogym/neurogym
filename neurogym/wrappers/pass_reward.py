import numpy as np
from gymnasium import Wrapper

from neurogym.utils import spaces


class PassReward(Wrapper):
    metadata: dict[str, str | None] = {  # noqa: RUF012
        "description": "Modifies observation by adding the previous reward.",
        "paper_link": None,
        "paper_name": None,
    }

    def __init__(self, env) -> None:
        """Modifies observation by adding the previous reward."""
        super().__init__(env)
        if isinstance(env.observation_space, spaces.Discrete):
            env_oss = env.observation_space.n  # Number of discrete states
            self.observation_space = spaces.Discrete(n=env_oss + 1)
        else:
            env_oss = env.observation_space.shape[0]
            self.observation_space = spaces.Box(
                -np.inf,
                np.inf,
                shape=(int(env_oss) + 1,),
                dtype=np.float32,
            )

    def reset(self, seed=None, options=None):
        step_fn = options.get("step_fn") if options else None
        if step_fn is None:
            step_fn = self.step
        return self.env.reset(options={"step_fn": step_fn}, seed=seed)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if isinstance(obs, np.ndarray):
            obs = np.concatenate((obs, np.array([reward])))
        else:
            obs = np.concatenate((np.array([obs]), np.array([reward])))
        return obs, reward, terminated, truncated, info
