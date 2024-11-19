import numpy as np

import neurogym as ngym
from neurogym import spaces


class DelayPairedAssociation(ngym.TrialEnv):
    """Delayed paired-association task.

    The agent is shown a pair of two stimuli separated by a delay period. For
    half of the stimuli-pairs shown, the agent should choose the Go response.
    The agent is rewarded if it chose the Go response correctly.
    """

    metadata = {  # noqa: RUF012
        "paper_link": "https://elifesciences.org/articles/43191",
        "paper_name": "Active information maintenance in working memory by a sensory cortex",
        "tags": ["perceptual", "working memory", "go-no-go", "supervised"],
    }

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0) -> None:
        super().__init__(dt=dt)
        self.choices = [0, 1]
        # trial conditions
        self.pairs = [(1, 3), (1, 4), (2, 3), (2, 4)]
        self.association = 0  # GO if np.diff(self.pair)[0]%2==self.association
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        # Durations (stimulus duration will be drawn from an exponential)

        # Rewards
        self.rewards = {"abort": -0.1, "correct": +1.0, "fail": -1.0, "miss": 0.0}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            "fixation": 0,
            "stim1": 1000,
            "delay_btw_stim": 1000,
            "stim2": 1000,
            "delay_aft_stim": 1000,
            "decision": 500,
        }
        if timing:
            self.timing.update(timing)

        self.abort = False
        # action and observation spaces
        name = {"fixation": 0, "stimulus": range(1, 5)}
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(5,),
            dtype=np.float32,
            name=name,
        )

        self.action_space = spaces.Discrete(2, name={"fixation": 0, "go": 1})

    def _new_trial(self, **kwargs):
        pair = self.pairs[self.rng.choice(len(self.pairs))]
        trial = {
            "pair": pair,
            "ground_truth": int(np.diff(pair)[0] % 2 == self.association),
        }
        trial.update(kwargs)
        pair = trial["pair"]

        periods = [
            "fixation",
            "stim1",
            "delay_btw_stim",
            "stim2",
            "delay_aft_stim",
            "decision",
        ]
        self.add_period(periods)

        # set observations
        self.add_ob(1, where="fixation")
        self.add_ob(1, "stim1", where=pair[0])
        self.add_ob(1, "stim2", where=pair[1])
        self.set_ob(0, "decision")
        # set ground truth
        self.set_groundtruth(trial["ground_truth"], "decision")

        # if trial is GO the reward is set to R_MISS and  to 0 otherwise
        self.r_tmax = self.rewards["miss"] * trial["ground_truth"]
        self.performance = 1 - trial["ground_truth"]

        return trial

    def _step(self, action, **kwargs):
        new_trial = False
        terminated = False
        truncated = False
        # rewards
        reward = 0
        ob = self.ob_now
        gt = self.gt_now
        # observations
        if self.in_period("fixation"):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards["abort"]
        elif self.in_period("decision") and action != 0:
            if action == gt:
                reward = self.rewards["correct"]
                self.performance = 1
            else:
                reward = self.rewards["fail"]
                self.performance = 0
            new_trial = True

        return ob, reward, terminated, truncated, {"new_trial": new_trial, "gt": gt}
