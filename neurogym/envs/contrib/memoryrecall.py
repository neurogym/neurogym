"""A generic memory recall task."""

from collections import OrderedDict

import numpy as np
from gymnasium import spaces

import neurogym as ngym


class MemoryRecall(ngym.TrialEnv):
    """Memory Recall.

    Args:
        stim_dim: int, stimulus dimension
        store_signal_dim: int, storage signal dimension
        t: int, sequence length
        p_recall: proportion of patterns stored for recall
        chance: chance level performance.
    """

    # TODO: Need to be made more general by passing the memories
    def __init__(
        self,
        dt=1,
        stim_dim=10,
        store_signal_dim=1,
        t_min=15,
        t_max=25,
        t_distribution="uniform",
        p_recall=0.1,
        chance=0.7,
        balanced=True,
        **kwargs,
    ) -> None:
        super().__init__(dt=dt)

        self.stim_dim = stim_dim
        self.store_signal_dim = store_signal_dim
        self.input_dim = self.stim_dim + self.store_signal_dim
        self.output_dim = stim_dim
        self.t_min = t_min
        if t_max is None:
            self.t_max = t_min
        else:
            self.t_max = t_max
        if self.t_max < self.t_min:
            msg = f"{t_max=} must be larger than {t_min=}."
            raise ValueError(msg)
        if t_distribution != "uniform":
            msg = f"{t_distribution=} only accepts 'uniform'."
            raise ValueError(msg)

        self.t_distribution = t_distribution
        self.generate_T = lambda: self.rng.randint(self.t_min, self.t_max + 1)
        self.p_recall = p_recall
        self.balanced = balanced
        self.chance = chance
        if self.balanced:
            self.p_unknown = (1 - chance) * 2.0
        else:
            # p_flip: amount of sensory noise during recall
            self.p_flip = 1 - chance

        if p_recall > 0.5:
            print("Cannot have p_recall larger than 0.5")

        # Environment specific
        self.action_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(stim_dim,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(stim_dim + 1,),
            dtype=np.float32,
        )

    def __str__(self) -> str:
        print("Recall dataset:")
        nicename_dict = OrderedDict(
            [
                ("stim_dim", "Stimulus dimension"),
                ("store_signal_dim", "Storage signal dimension"),
                ("input_dim", "Input dimension"),
                ("output_dim", "Output dimension"),
                ("t_min", "Minimum sequence length"),
                ("t_max", "Maximum sequence length"),
                ("t_distribution", "Sequence length distribution"),
                ("p_recall", "Proportion of recall"),
                ("chance", "Chancel level accuracy"),
            ],
        )
        if self.balanced:
            nicename_dict["p_unknown"] = "Proportion of unknown elements at recall"
        else:
            nicename_dict["p_flip"] = "Proportion of flipping at recall"

        string = ""
        for key, name in nicename_dict.items():
            string += f"{name} : {getattr(self, key)}\n"
        return string

    def _new_trial(self, **kwargs):
        # TODO: Need to be updated
        stim_dim = self.stim_dim

        t = self.generate_T()

        t_recall = int(self.p_recall * t)
        t_store = t - t_recall

        x_stim = np.zeros((t, stim_dim))
        x_store_signal = np.zeros((t, 1))
        y = np.zeros((t, stim_dim))
        m = np.zeros(t)

        # Storage phase
        if self.balanced:
            x_stim[:t_store, :] = (self.rng.rand(t_store, stim_dim) > 0.5) * 2.0 - 1.0
        else:
            x_stim[:t_store, :] = (self.rng.rand(t_store, stim_dim) > 0.5) * 1.0

        store_signal = self.rng.choice(np.arange(t_store), t_recall, replace=False)
        x_store_signal[store_signal, 0] = 1.0

        # Recall phase
        x_stim_recall = x_stim[store_signal]
        y[t_store:, :] = x_stim_recall
        m[t_store:] = 1.0

        # Perturb x_stim_recall
        # Flip probability
        if self.balanced:
            known_matrix = (self.rng.rand(t_recall, stim_dim) > self.p_unknown) * 1.0
            x_stim[t_store:, :stim_dim] = x_stim_recall * known_matrix
        else:
            flip_matrix = self.rng.rand(t_recall, stim_dim) < self.p_flip
            x_stim[t_store:, :stim_dim] = x_stim_recall * (1 - flip_matrix) + (1 - x_stim_recall) * flip_matrix

        x = np.concatenate((x_stim, x_store_signal), axis=1)
        self.ob = x
        self.gt = y
        self.mask = m
        self.tmax = t * self.dt

        return x, y, m

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and observations
        # ---------------------------------------------------------------------
        obs = self.ob_now
        gt = self.gt_now
        reward = np.mean(abs(gt - action)) * self.mask[self.t_ind]
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {"new_trial": False, "gt": gt}
