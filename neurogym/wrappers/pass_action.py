import numpy as np
from gymnasium import Wrapper, spaces


class PassAction(Wrapper):
    """Modifies observation by adding the previous action."""

    metadata: dict[str, str | None] = {  # noqa: RUF012
        "description": "Modifies observation by adding the previous action.",
        "paper_link": None,
        "paper_name": None,
    }

    def __init__(self, env) -> None:
        super().__init__(env)
        self.env = env
        # TODO: This is not adding one-hot
        env_oss = env.observation_space.shape[0]
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(env_oss + 1,),
            dtype=np.float32,
        )

    def reset(self, options=None):
        step_fn = options.get("step_fn") if options else None
        if step_fn is None:
            step_fn = self.step
        return self.env.reset(options={"step_fn": step_fn})

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = np.concatenate((obs, np.array([action])))
        return obs, reward, terminated, truncated, info
