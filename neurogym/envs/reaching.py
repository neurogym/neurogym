"""Reaching to target."""

import numpy as np

import neurogym as ngym
from neurogym import spaces
from neurogym.utils import tasktools


# TODO: Ground truth and action have different space,
# making it difficult for SL and RL to work together
# TODO: Need to clean up this task
class Reaching1D(ngym.TrialEnv):
    """Reaching to the stimulus.

    The agent is shown a stimulus during the fixation period. The stimulus
    encodes a one-dimensional variable such as a movement direction. At the
    end of the fixation period, the agent needs to respond by reaching
    towards the stimulus direction.
    """

    metadata = {  # noqa: RUF012
        "paper_link": "https://science.sciencemag.org/content/233/4771/1416",
        "paper_name": "Neuronal population coding of movement direction",
        "tags": ["motor", "steps action space"],
    }

    def __init__(self, dt=100, rewards=None, timing=None, dim_ring=16) -> None:
        super().__init__(dt=dt)
        # Rewards
        self.rewards = {"correct": +1.0, "fail": -0.1}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {"fixation": 500, "reach": 500}
        if timing:
            self.timing.update(timing)

        # action and observation spaces
        obs_name = {"self": range(dim_ring, 2 * dim_ring), "target": range(dim_ring)}
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(2 * dim_ring,),
            dtype=np.float32,
            name=obs_name,
        )
        action_name = {"fixation": 0, "left": 1, "right": 2}
        self.action_space = spaces.Discrete(3, name=action_name)

        self.theta = np.arange(0, 2 * np.pi, 2 * np.pi / dim_ring)
        self.state = np.pi
        self.dim_ring = dim_ring

    def _new_trial(self, **kwargs):
        # Trial
        self.state = np.pi
        trial = {"ground_truth": self.rng.uniform(0, np.pi * 2)}
        trial.update(kwargs)

        # Periods
        self.add_period(["fixation", "reach"])

        target = np.cos(self.theta - trial["ground_truth"])
        self.add_ob(target, "reach", where="target")

        self.set_groundtruth(np.pi, "fixation")
        self.set_groundtruth(trial["ground_truth"], "reach")
        self.dec_per_dur = self.end_ind["reach"] - self.start_ind["reach"]

        return trial

    def _step(self, action):
        terminated = False
        truncated = False
        if action == 1:
            self.state += 0.05
        elif action == 2:
            self.state -= 0.05

        self.state = np.mod(self.state, 2 * np.pi)

        gt = self.gt_now
        if self.in_period("fixation"):
            reward = 0
        else:
            reward = np.max(
                (
                    self.rewards["correct"] - tasktools.circular_dist(self.state - gt),
                    self.rewards["fail"],
                ),
            )
            norm_rew = (reward - self.rewards["fail"]) / (self.rewards["correct"] - self.rewards["fail"])
            self.performance += norm_rew / self.dec_per_dur

        return self.ob_now, reward, terminated, truncated, {"new_trial": False}

    def post_step(self, ob, reward, terminated, truncated, info):
        """Modify observation."""
        ob[self.dim_ring :] = np.cos(self.theta - self.state)
        return ob, reward, terminated, truncated, info


class Reaching1DWithSelfDistraction(ngym.TrialEnv):
    """Reaching with self distraction.

    In this task, the reaching state itself generates strong inputs that
    overshadows the actual target input. This task is inspired by behavior
    in electric fish where the electric sensing organ is distracted by
    discharges from its own electric organ for active sensing.
    Similar phenomena in bats.
    """

    metadata = {  # noqa: RUF012
        "description": """The agent has to reproduce the angle indicated
         by the observation. Furthermore, the reaching state itself
         generates strong inputs that overshadows the actual target input.""",
        "paper_link": None,
        "paper_name": None,
        "tags": ["motor", "steps action space"],
    }

    def __init__(self, dt=100, rewards=None, timing=None) -> None:
        super().__init__(dt=dt)
        # Rewards
        self.rewards = {"correct": +1.0, "fail": -0.1}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {"fixation": 500, "reach": 500}
        if timing:
            self.timing.update(timing)

        # action and observation spaces
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(32,),
            dtype=np.float32,
        )
        self.theta = np.arange(0, 2 * np.pi, 2 * np.pi / 32)
        self.state = np.pi

    def _new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        self.state = np.pi
        trial = {"ground_truth": self.rng.uniform(0, np.pi * 2)}
        trial.update(kwargs)
        # ---------------------------------------------------------------------
        # Periods
        # ---------------------------------------------------------------------
        self.add_period("fixation")
        self.add_period("reach", after="fixation")

        ob = self.view_ob("reach")
        # Signal is weaker than the self-distraction
        ob += np.cos(self.theta - trial["ground_truth"]) * 0.3

        self.set_groundtruth(np.pi, "fixation")
        self.set_groundtruth(trial["ground_truth"], "reach")
        self.dec_per_dur = self.end_ind["reach"] - self.start_ind["reach"]

        return trial

    def _step(self, action):
        terminated = False
        truncated = False
        if action == 1:
            self.state += 0.05
        elif action == 2:
            self.state -= 0.05
        self.state = np.mod(self.state, 2 * np.pi)

        gt = self.gt_now
        if self.in_period("fixation"):
            reward = 0
        else:
            reward = np.max(
                (
                    self.rewards["correct"] - tasktools.circular_dist(self.state - gt),
                    self.rewards["fail"],
                ),
            )
            norm_rew = (reward - self.rewards["fail"]) / (self.rewards["correct"] - self.rewards["fail"])
            self.performance += norm_rew / self.dec_per_dur

        return self.ob_now, reward, terminated, truncated, {"new_trial": False}

    def post_step(self, ob, reward, terminated, truncated, info):
        """Modify observation."""
        ob += np.cos(self.theta - self.state)
        return ob, reward, terminated, truncated, info
