"""Created on Mon Jan 27 11:00:26 2020.

@author: martafradera
"""

import warnings

import numpy as np

import neurogym as ngym
from neurogym import spaces
from neurogym.utils.ngym_random import TruncExp


class Detection(ngym.TrialEnv):
    """The agent has to GO if a stimulus is presented.

    Args:
        delay: If not None indicates the delay, from the moment of the start of
            the stimulus period when the actual stimulus is presented. Otherwise,
            the delay is drawn from a uniform distribution. (def: None (ms), int)
        stim_dur: Stimulus duration. (def: 100 (ms), int)
    """

    # TODO: Remains to be described.
    metadata = {  # noqa: RUF012
        "paper_link": None,
        "paper_name": None,
        "tags": ["perceptual", "reaction time", "go-no-go", "supervised"],
    }

    def __init__(
        self,
        dt=100,
        rewards=None,
        timing=None,
        sigma=1.0,
        delay=None,
        stim_dur=100,
    ) -> None:
        super().__init__(dt=dt)
        # Possible decisions at the end of the trial
        self.choices = [0, 1]

        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        self.delay = delay
        self.stim_dur = int(stim_dur / self.dt)  # in steps should be greater
        # than 1 stp else it wont have enough time to respond within the window
        if self.stim_dur == 1:
            self.extra_step = 1
            if delay is None:
                warnings.warn(
                    "Added an extra stp after the actual stimulus, else model will not be able to respond "
                    "within response window (stimulus epoch).",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            self.extra_step = 0

        if self.stim_dur < 1:
            warnings.warn("Stimulus duration shorter than dt", stacklevel=2)

        # Rewards
        self.rewards = {"abort": -0.1, "correct": +1.0, "fail": -1.0, "miss": -1}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            "fixation": 500,
            "stimulus": TruncExp(1000, 500, 1500),
        }
        if timing:
            self.timing.update(timing)

        # whether to abort (T) or not (F) the trial when breaking fixation:
        self.abort = False

        name = {"fixation": 0, "stimulus": 1}
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(2,),
            dtype=np.float32,
            name=name,
        )

        self.action_space = spaces.Discrete(2, name={"fixation": 0, "go": 1})

    def _new_trial(self, **kwargs):
        # Trial
        trial = {"ground_truth": self.rng.choice(self.choices)}
        trial.update(kwargs)  # allows wrappers to modify the trial
        ground_truth = trial["ground_truth"]

        # Period
        self.add_period(["fixation", "stimulus"])

        # Observations
        # TODO: The design of this task appears unnecessarily complicated
        # all observation values are 0 by default
        # FIXATION: setting fixation cue to 1 during fixation period
        self.set_ob(1, "fixation", where="fixation")
        # stimulus:
        stim = self.view_ob("stimulus")
        stim[:, 1:] += self.rng.randn(stim.shape[0], 1) * self.sigma
        # delay
        # SET THE STIMULUS
        # adding gaussian noise to stimulus with std = self.sigma
        if ground_truth == 1:
            if self.delay is None:
                # there must be a step after the stim or the model will not
                # be able to respond on time
                max_delay = stim.shape[0] - self.stim_dur - self.extra_step
                delay = 0 if max_delay == 0 else self.rng.randint(0, max_delay)
            else:
                delay = self.delay
            stim[delay : delay + self.stim_dur, 1] += 0.5  # actual stim
            self.r_tmax = self.rewards["miss"]
            self.performance = 0
        else:
            stim[:, 1:] += self.rng.randn(stim.shape[0], 1) * self.sigma
            delay = 0
            self.r_tmax = 0  # response omision is correct but not rewarded
            self.performance = 1

        self.delay_trial = delay * self.dt

        # Set ground-truth
        if ground_truth == 1:
            dec = self.view_groundtruth("stimulus")
            dec[delay:] = 1

        return trial

    def _step(self, action):
        new_trial = False
        terminated = False
        truncated = False
        # rewards
        reward = 0
        gt = self.gt_now
        # Example structure
        if self.in_period("fixation"):  # during fixation period
            if action != 0:  # if fixation break
                new_trial = self.abort
                reward = self.rewards["abort"]
        elif self.in_period("stimulus") and action != 0:  # original
            new_trial = True
            if (action == self.trial["ground_truth"]) and (self.t >= self.end_t["fixation"] + self.delay_trial):
                reward = self.rewards["correct"]
                self.performance = 1
            else:  # if incorrect
                reward = self.rewards["fail"]
                self.performance = 0

        return (
            self.ob_now,
            reward,
            terminated,
            truncated,
            {"new_trial": new_trial, "gt": gt},
        )
