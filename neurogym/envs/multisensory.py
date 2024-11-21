"""Multi-Sensory Integration."""

import numpy as np

import neurogym as ngym
from neurogym import spaces


# TODO: This is not finished yet. Need to compare with original paper
# TODO: In this current implementation, the two stimuli always point to the
#  same direction, check original
class MultiSensoryIntegration(ngym.TrialEnv):
    """Multi-sensory integration.

    Two stimuli are shown in two input modalities. Each stimulus points to
    one of the possible responses with a certain strength (coherence). The
    correct choice is the response with the highest summed strength from
    both stimuli. The agent is therefore encouraged to integrate information
    from both modalities equally.
    """

    metadata = {  # noqa: RUF012
        "description": None,
        "paper_link": None,
        "paper_name": None,
        "tags": ["perceptual", "two-alternative", "supervised"],
    }

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, dim_ring=2) -> None:
        super().__init__(dt=dt)

        # trial conditions
        self.cohs = [5, 15, 50]

        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {"abort": -0.1, "correct": +1.0}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {"fixation": 300, "stimulus": 750, "decision": 100}
        if timing:
            self.timing.update(timing)
        self.abort = False

        # set action and observation space
        self.theta = np.linspace(0, 2 * np.pi, dim_ring + 1)[:-1]
        self.choices = np.arange(dim_ring)

        name = {
            "fixation": 0,
            "stimulus_mod1": range(1, dim_ring + 1),
            "stimulus_mod2": range(dim_ring + 1, 2 * dim_ring + 1),
        }
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(1 + 2 * dim_ring,),
            dtype=np.float32,
            name=name,
        )

        name = {"fixation": 0, "choice": range(1, dim_ring + 1)}
        self.action_space = spaces.Discrete(1 + dim_ring, name=name)

    def _new_trial(self, **kwargs):
        # Trial info
        trial = {
            "ground_truth": self.rng.choice(self.choices),
            "coh": self.rng.choice(self.cohs),
            "coh_prop": self.rng.rand(),
        }
        trial.update(kwargs)

        coh_0 = trial["coh"] * trial["coh_prop"]
        coh_1 = trial["coh"] * (1 - trial["coh_prop"])
        ground_truth = trial["ground_truth"]
        stim_theta = self.theta[ground_truth]

        # Periods
        periods = ["fixation", "stimulus", "decision"]
        self.add_period(periods)

        self.add_ob(1, where="fixation")
        stim = np.cos(self.theta - stim_theta) * (coh_0 / 200) + 0.5
        self.add_ob(stim, "stimulus", where="stimulus_mod1")
        stim = np.cos(self.theta - stim_theta) * (coh_1 / 200) + 0.5
        self.add_ob(stim, "stimulus", where="stimulus_mod2")
        self.add_randn(0, self.sigma, "stimulus")
        self.set_ob(0, "decision")

        self.set_groundtruth(ground_truth, period="decision", where="choice")

        return trial

    def _step(self, action):
        ob = self.ob_now
        gt = self.gt_now

        new_trial = False
        terminated = False
        truncated = False
        reward = 0
        if self.in_period("fixation"):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards["abort"]
        elif self.in_period("decision") and action != 0:  # broke fixation
            new_trial = True
            if action == gt:
                reward = self.rewards["correct"]
                self.performance = 1

        return ob, reward, terminated, truncated, {"new_trial": new_trial, "gt": gt}
