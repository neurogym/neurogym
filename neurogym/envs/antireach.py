"""Anti-reach or anti-saccade task."""

import numpy as np

import neurogym as ngym
from neurogym import spaces


class AntiReach(ngym.TrialEnv):
    """Anti-response task.

    During the fixation period, the agent fixates on a fixation point.
    During the following stimulus period, the agent is then shown a stimulus away
    from the fixation point. Finally, the agent needs to respond in the
    opposite direction of the stimulus during the decision period.

    Args:
        anti: bool, if True, requires an anti-response. If False, requires a
            pro-response, i.e. response towards the stimulus.
    """

    metadata = {  # noqa: RUF012
        "paper_link": "https://www.nature.com/articles/nrn1345",
        "paper_name": """Look away: the anti-saccade task and
        the voluntary control of eye movement""",
        "tags": ["perceptual", "steps action space"],
    }

    def __init__(self, dt=100, anti=True, rewards=None, timing=None, dim_ring=32) -> None:
        super().__init__(dt=dt)

        self.anti = anti

        # Rewards
        self.rewards = {"abort": -0.1, "correct": +1.0, "fail": 0.0}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {"fixation": 500, "stimulus": 500, "delay": 0, "decision": 500}
        if timing:
            self.timing.update(timing)

        self.abort = False

        # action and observation spaces
        self.dim_ring = dim_ring
        self.theta = np.arange(0, 2 * np.pi, 2 * np.pi / dim_ring)
        self.choices = np.arange(dim_ring)

        name = {"fixation": 0, "stimulus": range(1, dim_ring + 1)}
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(1 + dim_ring,),
            dtype=np.float32,
            name=name,
        )

        name = {"fixation": 0, "choice": range(1, dim_ring + 1)}
        self.action_space = spaces.Discrete(1 + dim_ring, name=name)

    def _new_trial(self, **kwargs):
        # Trial info
        trial = {
            "ground_truth": self.rng.choice(self.choices),
            "anti": self.anti,
        }
        trial.update(kwargs)

        ground_truth = trial["ground_truth"]
        stim_theta = np.mod(self.theta[ground_truth] + np.pi, 2 * np.pi) if trial["anti"] else self.theta[ground_truth]

        # Periods
        periods = ["fixation", "stimulus", "delay", "decision"]
        self.add_period(periods)

        self.add_ob(1, period=["fixation", "stimulus", "delay"], where="fixation")
        stim = np.cos(self.theta - stim_theta)
        self.add_ob(stim, "stimulus", where="stimulus")

        self.set_groundtruth(ground_truth, period="decision", where="choice")

        return trial

    def _step(self, action):
        new_trial = False
        terminated = False
        truncated = False
        # rewards
        reward = 0
        gt = self.gt_now
        # observations
        if self.in_period("fixation"):
            if action != 0:  # action = 0 means fixating
                new_trial = self.abort
                reward += self.rewards["abort"]
        elif self.in_period("decision") and action != 0:
            new_trial = True
            if action == gt:
                reward += self.rewards["correct"]
                self.performance = 1
            else:
                reward += self.rewards["fail"]

        return (
            self.ob_now,
            reward,
            terminated,
            truncated,
            {"new_trial": new_trial, "gt": gt},
        )
