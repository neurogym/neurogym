from __future__ import annotations

from typing import TYPE_CHECKING

from gymnasium import Wrapper

if TYPE_CHECKING:
    from neurogym.core import TrialEnv


class Noise(Wrapper):
    """Add Gaussian noise to the observations.

    Args:
        env: The NeuroGym environment to wrap.
        std_noise: Standard deviation of noise. (def: 0.1)
        perf_th: If != None, the wrapper will adjust the noise so the mean
            performance is not larger than perf_th. (def: None, float)
        w: Window used to compute the mean performance. (def: 100, int)
        step_noise: Step used to increment/decrease std. (def: 0.001, float)
    """

    metadata: dict[str, str | None] = {  # noqa: RUF012
        "description": "Add Gaussian noise to the observations.",
        "paper_link": None,
        "paper_name": None,
    }

    def __init__(
        self,
        env: TrialEnv,
        std_noise: float = 0.1,
    ) -> None:
        super().__init__(env)
        self.env = env
        self.std_noise = std_noise

    def reset(self, options=None):
        step_fn = options.get("step_fn") if options else None
        if step_fn is None:
            step_fn = self.step
        return self.env.reset(options={"step_fn": step_fn})

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # add noise
        obs += self.env.rng.normal(loc=0, scale=self.std_noise, size=obs.shape)
        return obs, reward, terminated, truncated, info
