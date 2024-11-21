import numpy as np
from gymnasium import spaces

import neurogym as ngym

# TODO: are trials counted correctly in this task?

# TODO: No longer maintained, use existing task to build this task


class IBL(ngym.TrialEnv):
    metadata = {  # noqa: RUF012
        "paper_link": "https://www.sciencedirect.com/science/article/pii/S0896627317311364",
        "paper_name": """An International Laboratory for Systems and ' +
        'Computational Neuroscience""",
    }

    def __init__(self, dt: int = 100, rewards: dict[str, float] | None = None) -> None:
        super().__init__(dt=dt)
        # Fix: Use numpy's default_rng instead of RandomState
        self._rng = np.random.default_rng(0)
        self.sigma = 0.10  # noise
        self.num_tr = 0  # number of trials
        self.block = 0  # block id
        self.block_size = 10000

        # Rewards
        self.rewards = {"correct": +1, "fail": 0.0}
        if rewards:
            self.rewards.update(rewards)

        # trial conditions (left, right)
        self.choices = [0, 1]

        self.cohs = np.array([1.6, 3.2, 6.4, 12.8, 25.6, 51.2])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(2,),
            dtype=np.float32,
        )
        self.observation_shape = self.observation_space.shape  # Add this line

    def new_block(self, n_trial: int, probs: tuple[float, float] | None = None) -> None:
        if probs is None:
            probs = (0.5, 0.5)  # Default probabilities if none are provided
        self.ground_truth = self._rng.choice(self.choices, size=(n_trial,), p=probs)
        self.coh = self._rng.choice(self.cohs, size=(n_trial,))

        obs = np.zeros((n_trial, int(self.observation_shape[0])))  # Use self.observation_shape
        ind = np.arange(n_trial)
        obs[ind, self.ground_truth] = 0.5 + self.coh / 200
        obs[ind, 1 - self.ground_truth] = 0.5 - self.coh / 200

        # Add observation noise
        obs += self._rng.standard_normal(obs.shape) * self.sigma
        self.ob = obs

    def _new_trial(self, **kwargs) -> None:  # type: ignore[override]
        """Called when a trial ends to get the specifications of the next trial.

        Such specifications are stored in a dictionary with the following items:
            durations, which stores the duration of the different periods (in
            the case of perceptualDecisionMaking: fixation, stimulus and decision periods)
            ground truth: correct response for the trial
            coh: stimulus coherence (evidence) for the trial.
        """
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        self.ind = self.num_tr % self.block_size
        if self.ind == 0:
            self.new_block(self.block_size)

        self.num_tr += 1

    def _step(self) -> tuple:  # type: ignore[override]
        info = {
            "continue": True,
            "gt": self.ground_truth[self.ind],
            "coh": self.coh[self.ind],
            "block": self.block,
        }
        obs = self.ob[self.ind]

        # reward of last trial
        reward = self.rewards["correct"]  # TODO: need to be done

        # ---------------------------------------------------------------------
        # new trial?
        info["new_trial"] = True
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, info


class IBL_Block(IBL):  # noqa: N801
    def __init__(self, dt: int = 100) -> None:
        super().__init__(dt=dt)
        self.probs = ((0.2, 0.8), (0.8, 0.2), (0.5, 0.5))
        self.block = 0
        self.block_size = 200

    def _new_trial(self, **kwargs) -> None:  # type: ignore[override]
        """Called when a trial ends to get the specifications of the next trial.

        Such specifications are stored in a dictionary with the following items:
            durations, which stores the duration of the different periods (in
            the case of perceptualDecisionMaking: fixation, stimulus and decision periods)
            ground truth: correct response for the trial
            coh: stimulus coherence (evidence) for the trial.
        """
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        self.ind = self.num_tr % self.block_size
        if self.ind == 0:
            self.block = self._rng.integers(0, 3)  # Choose 0, 1, or 2
            prob = self.probs[self.block]
            self.new_block(self.block_size, probs=prob)

        self.num_tr += 1
