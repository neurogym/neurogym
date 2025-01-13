"""Hierarchical reasoning tasks."""

import numpy as np

import neurogym as ngym
from neurogym import spaces
from neurogym.utils.ngym_random import TruncExp


class HierarchicalReasoning(ngym.TrialEnv):
    """Hierarchical reasoning of rules.

    On each trial, the subject receives two flashes separated by a delay
    period. The subject needs to judge whether the duration of this delay
    period is shorter than a threshold. Both flashes appear at the
    same location on each trial. For one trial type, the network should
    report its decision by going to the location of the flashes if the delay is
    shorter than the threshold. In another trial type, the network should go to
    the opposite direction of the flashes if the delay is short.
    The two types of trials are alternated across blocks, and the block
    transtion is unannouced.
    """

    metadata = {  # noqa: RUF012
        "paper_link": "https://science.sciencemag.org/content/364/6441/eaav8911",
        "paper_name": "Hierarchical reasoning by neural circuits in the frontal cortex",
        "tags": ["perceptual", "two-alternative", "supervised"],
    }

    def __init__(self, dt=100, rewards=None, timing=None) -> None:
        super().__init__(dt=dt)
        self.choices = [0, 1]

        self.rewards = {"abort": -0.1, "correct": +1.0, "fail": 0.0}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            "fixation": TruncExp(600, 400, 800),
            "rule_target": 1000,
            "fixation2": TruncExp(600, 400, 900),
            "flash1": 100,
            "delay": (530, 610, 690, 770, 850, 930, 1010, 1090, 1170),
            "flash2": 100,
            "decision": 700,
        }
        if timing:
            self.timing.update(timing)
        self.mid_delay = np.median(self.timing["delay"][1])

        self.abort = False

        name = {"fixation": 0, "rule": [1, 2], "stimulus": [3, 4]}
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(5,),
            dtype=np.float32,
            name=name,
        )
        name = {"fixation": 0, "rule": [1, 2], "choice": [3, 4]}
        self.action_space = spaces.Discrete(5, name=name)

        self.chose_correct_rule = False
        self.rule = 0
        self.trial_in_block = 0
        self.block_size = 10
        self.new_block()

    def new_block(self) -> None:
        self.block_size = self.rng.randint(10, 20 + 1)
        self.rule = 1 - self.rule  # alternate rule
        self.trial_in_block = 0

    def _new_trial(self, **kwargs):
        interval = self.sample_time("delay")
        trial = {
            "interval": interval,
            "rule": self.rule,
            "stimulus": self.rng.choice(self.choices),
        }
        trial.update(kwargs)

        # Is interval long? When interval == mid_delay, randomly assign
        long_interval = interval > self.mid_delay + (self.rng.rand() - 0.5)
        # Is the response pro or anti?
        pro_choice = int(long_interval) == trial["rule"]
        trial["long_interval"] = long_interval
        trial["pro_choice"] = pro_choice

        # Periods
        periods = [
            "fixation",
            "rule_target",
            "fixation2",
            "flash1",
            "delay",
            "flash2",
            "decision",
        ]
        self.add_period(periods)

        # Observations
        stimulus = self.observation_space.name["stimulus"][trial["stimulus"]]
        choice = trial["stimulus"] if pro_choice else 1 - trial["stimulus"]

        self.add_ob(1, where="fixation")
        self.set_ob(0, "decision", where="fixation")
        self.add_ob(1, "rule_target", where="rule")
        self.add_ob(1, "flash1", where=stimulus)
        self.add_ob(1, "flash2", where=stimulus)

        # Ground truth
        self.set_groundtruth(choice, period="decision", where="choice")
        self.set_groundtruth(trial["rule"], period="rule_target", where="rule")

        # Start new block?
        self.trial_in_block += 1
        if self.trial_in_block >= self.block_size:
            self.new_block()

        return trial

    def _step(self, action):
        new_trial = False
        terminated = False
        truncated = False
        # rewards
        reward = 0
        gt = self.gt_now
        # observations
        if self.in_period("decision"):
            if action != 0:
                new_trial = True
                if (action == gt) and self.chose_correct_rule:
                    reward += self.rewards["correct"]
                    self.performance = 1
                else:
                    reward += self.rewards["fail"]
        elif self.in_period("rule_target"):
            self.chose_correct_rule = action == gt
        elif action != 0:  # action = 0 means fixating
            new_trial = self.abort
            reward += self.rewards["abort"]

        if new_trial:
            self.chose_correct_rule = False

        return (
            self.ob_now,
            reward,
            terminated,
            truncated,
            {"new_trial": new_trial, "gt": gt},
        )
