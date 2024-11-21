import numpy as np

import neurogym as ngym
from neurogym import spaces


# TODO: Getting duration is not intuitive, not clear to people
class IntervalDiscrimination(ngym.TrialEnv):
    """Comparing the time length of two stimuli.

    Two stimuli are shown sequentially, separated by a delay period. The
    duration of each stimulus is randomly sampled on each trial. The
    subject needs to judge which stimulus has a longer duration, and reports
    its decision during the decision period by choosing one of the two
    choice options.
    """

    metadata = {  # noqa: RUF012
        "paper_link": "https://www.sciencedirect.com/science/article/pii/S0896627309004887",
        "paper_name": """Feature- and Order-Based Timing Representations
         in the Frontal Cortex""",
        "tags": [
            "timing",
            "working memory",
            "delayed response",
            "two-alternative",
            "supervised",
        ],
    }

    def __init__(self, dt=80, rewards=None, timing=None) -> None:
        super().__init__(dt=dt)
        # Rewards
        self.rewards = {"abort": -0.1, "correct": +1.0, "fail": 0.0}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            "fixation": 300,
            "stim1": lambda: self.rng.uniform(300, 600),
            "delay1": lambda: self.rng.uniform(800, 1500),
            "stim2": lambda: self.rng.uniform(300, 600),
            "delay2": 500,
            "decision": 300,
        }
        if timing:
            self.timing.update(timing)

        self.abort = False

        name = {"fixation": 0, "stim1": 1, "stim2": 2}
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(3,),
            dtype=np.float32,
            name=name,
        )
        name = {"fixation": 0, "choice1": 1, "choice2": 2}
        self.action_space = spaces.Discrete(3, name=name)

    def _new_trial(self, **kwargs):
        duration1 = self.sample_time("stim1")
        duration2 = self.sample_time("stim2")
        ground_truth = 1 if duration1 > duration2 else 2
        trial = {
            "duration1": duration1,
            "duration2": duration2,
            "ground_truth": ground_truth,
        }

        periods = ["fixation", "stim1", "delay1", "stim2", "delay2", "decision"]
        durations = [None, duration1, None, duration2, None, None]
        self.add_period(periods, duration=durations)

        self.add_ob(1, where="fixation")
        self.add_ob(1, "stim1", where="stim1")
        self.add_ob(1, "stim2", where="stim2")
        self.set_ob(0, "decision")

        self.set_groundtruth(ground_truth, "decision")

        return trial

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
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
                reward = self.rewards["abort"]
        elif self.in_period("decision") and action != 0:
            new_trial = True
            if action == gt:
                reward = self.rewards["correct"]
                self.performance = 1
            else:
                reward = self.rewards["fail"]

        return (
            self.ob_now,
            reward,
            terminated,
            truncated,
            {"new_trial": new_trial, "gt": gt},
        )
