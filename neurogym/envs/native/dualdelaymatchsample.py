import numpy as np

import neurogym as ngym
from neurogym import spaces


class DualDelayMatchSample(ngym.TrialEnv):
    """Two-item Delay-match-to-sample.

    The trial starts with a fixation period. Then during the sample period,
    two sample stimuli are shown simultaneously. Followed by the first delay
    period, a cue is shown, indicating which sample stimulus will be tested.
    Then the first test stimulus is shown and the agent needs to report whether
    this test stimulus matches the cued sample stimulus. Then another delay
    and then test period follows, and the agent needs to report whether the
    other sample stimulus matches the second test stimulus.
    """

    metadata = {  # noqa: RUF012
        "paper_link": "https://science.sciencemag.org/content/354/6316/1136",
        "paper_name": """Reactivation of latent working memories with
        transcranial magnetic stimulation""",
        "tags": ["perceptual", "working memory", "two-alternative", "supervised"],
    }

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0) -> None:
        super().__init__(dt=dt)
        self.choices = [1, 2]
        self.cues = [0, 1]

        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {"abort": -0.1, "correct": +1.0, "fail": 0.0}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            "fixation": 500,
            "sample": 500,
            "delay1": 500,
            "cue1": 500,
            "test1": 500,
            "delay2": 500,
            "cue2": 500,
            "test2": 500,
        }
        if timing:
            self.timing.update(timing)

        self.abort = False

        name = {
            "fixation": 0,
            "stimulus1": range(1, 3),
            "stimulus2": range(3, 5),
            "cue1": 5,
            "cue2": 6,
        }
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(7,),
            dtype=np.float32,
            name=name,
        )
        name = {"fixation": 0, "match": 1, "non-match": 2}
        self.action_space = spaces.Discrete(3, name=name)

    def _new_trial(self, **kwargs):
        trial = {
            "ground_truth1": self.rng.choice(self.choices),
            "ground_truth2": self.rng.choice(self.choices),
            "sample1": self.rng.choice([0, 0.5]),
            "sample2": self.rng.choice([0, 0.5]),
            "test_order": self.rng.choice([0, 1]),
        }
        trial.update(kwargs)

        ground_truth1 = trial["ground_truth1"]
        ground_truth2 = trial["ground_truth2"]
        sample1 = trial["sample1"]
        sample2 = trial["sample2"]

        test1 = sample1 if ground_truth1 == 1 else 0.5 - sample1
        test2 = sample2 if ground_truth2 == 1 else 0.5 - sample2
        trial["test1"] = test1
        trial["test2"] = test2

        if trial["test_order"] == 0:
            stim_test1_period, stim_test2_period = "test1", "test2"
            cue1_period, cue2_period = "cue1", "cue2"
        else:
            stim_test1_period, stim_test2_period = "test2", "test1"
            cue1_period, cue2_period = "cue2", "cue1"

        sample_theta, test_theta = sample1 * np.pi, test1 * np.pi
        stim_sample1 = [np.cos(sample_theta), np.sin(sample_theta)]
        stim_test1 = [np.cos(test_theta), np.sin(test_theta)]

        sample_theta, test_theta = sample2 * np.pi, test2 * np.pi
        stim_sample2 = [np.cos(sample_theta), np.sin(sample_theta)]
        stim_test2 = [np.cos(test_theta), np.sin(test_theta)]

        periods = [
            "fixation",
            "sample",
            "delay1",
            "cue1",
            "test1",
            "delay2",
            "cue2",
            "test2",
        ]
        self.add_period(periods)

        self.add_ob(1, where="fixation")
        self.add_ob(stim_sample1, "sample", where="stimulus1")
        self.add_ob(stim_sample2, "sample", where="stimulus2")
        self.add_ob(1, cue1_period, where="cue1")
        self.add_ob(1, cue2_period, where="cue2")
        self.add_ob(stim_test1, stim_test1_period, where="stimulus1")
        self.add_ob(stim_test2, stim_test2_period, where="stimulus2")
        self.add_randn(0, self.sigma, "sample")
        self.add_randn(0, self.sigma, "test1")
        self.add_randn(0, self.sigma, "test2")

        self.set_groundtruth(ground_truth1, stim_test1_period)
        self.set_groundtruth(ground_truth2, stim_test2_period)

        return trial

    def _step(self, action):
        new_trial = False
        terminated = False
        truncated = False
        reward = 0

        ob = self.ob_now
        gt = self.gt_now

        if self.in_period("test1"):
            if action != 0:
                if action == gt:
                    reward = self.rewards["correct"]
                    self.performance = 1
                else:
                    reward = self.rewards["fail"]
        elif self.in_period("test2"):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.rewards["correct"]
                    self.performance = 1
                else:
                    reward = self.rewards["fail"]
        elif action != 0:
            new_trial = self.abort
            reward = self.rewards["abort"]

        return ob, reward, terminated, truncated, {"new_trial": new_trial, "gt": gt}
