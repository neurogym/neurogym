"""Multi-arm Bandit task."""
# TODO: add the actual papers.

import numpy as np

import neurogym as ngym
from neurogym import spaces


class Bandit(ngym.TrialEnv):
    """Multi-arm bandit task.

    On each trial, the agent is presented with multiple choices. Each
    option produces a reward of a certain magnitude given a certain probability.

    Args:
        n: int, the number of choices (arms)
        p: tuple of length n, describes the probability of each arm
            leading to reward
        rewards: tuple of length n, describe the reward magnitude of each option when rewarded
    """

    metadata = {  # noqa: RUF012
        "paper_link": "https://www.nature.com/articles/s41593-018-0147-8",
        "paper_name": "Prefrontal cortex as a meta-reinforcement learning system",
        "tags": ["n-alternative"],
    }

    def __init__(
        self,
        dt: int = 100,
        n: int = 2,
        p: tuple[float, ...] | list[float] = (0.5, 0.5),
        rewards: list[float] | np.ndarray | None = None,
        timing: dict | None = None,
    ) -> None:
        super().__init__(dt=dt)
        if timing is not None:
            print("Warning: Bandit task does not require timing variable.")

        self.n = n
        self._p = np.array(p)  # Reward probabilities

        if rewards is not None:
            self._rewards = np.array(rewards)
        else:
            self._rewards = np.ones(n)  # 1 for every arm

        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(1,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(n)

    def _new_trial(self, **kwargs):
        # Create a new dictionary with NumPy arrays converted to lists
        trial = {"p": self._p.tolist(), "rewards": self._rewards.tolist()}
        trial.update(kwargs)

        self.ob = np.zeros((1, self.observation_space.shape[0]))

        return trial

    def _step(self, action):
        trial = self.trial
        terminated = False
        truncated = False

        ob = self.ob[0]
        p = np.array(trial["p"])
        rewards = np.array(trial["rewards"])
        reward = (self.rng.random() < p[action]) * rewards[action]
        info = {"new_trial": True}

        return ob, reward, terminated, truncated, info
