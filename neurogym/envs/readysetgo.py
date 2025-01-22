"""Ready-set-go task."""

import numpy as np

import neurogym as ngym
from neurogym import spaces
from neurogym.utils.ngym_random import TruncExp


class ReadySetGo(ngym.TrialEnv):
    """Agents have to measure and produce different time intervals.

    A stimulus is briefly shown during a ready period, then again during a
    set period. The ready and set periods are separated by a measure period,
    the duration of which is randomly sampled on each trial. The agent is
    required to produce a response after the set cue such that the interval
    between the response and the set cue is as close as possible to the
    duration of the measure period.

    Args:
        gain: Controls the measure that the agent has to produce. (def: 1, int)
        prod_margin: controls the interval around the ground truth production
            time within which the agent receives proportional reward
    """

    metadata = {  # noqa: RUF012
        "paper_link": "https://www.sciencedirect.com/science/article/pii/S0896627318304185",
        "paper_name": """Flexible Sensorimotor Computations through Rapid
        Reconfiguration of Cortical Dynamics""",
        "tags": ["timing", "go-no-go", "supervised"],
    }

    def __init__(self, dt=80, rewards=None, timing=None, gain=1, prod_margin=0.2) -> None:
        super().__init__(dt=dt)
        self.prod_margin = prod_margin

        self.gain = gain

        # Rewards
        self.rewards = {"abort": -0.1, "correct": +1.0, "fail": 0.0}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            "fixation": 100,
            "ready": 83,
            "measure": lambda: self.rng.uniform(800, 1500),
            "set": 83,
        }
        if timing:
            self.timing.update(timing)

        self.abort = False
        # set action and observation space
        name = {"fixation": 0, "ready": 1, "set": 2}
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(3,),
            dtype=np.float32,
            name=name,
        )

        name = {"fixation": 0, "go": 1}
        self.action_space = spaces.Discrete(2, name=name)  # (fixate, go)

    def _new_trial(self, **kwargs):
        measure = self.sample_time("measure")
        trial = {"measure": measure, "gain": self.gain}
        trial.update(kwargs)

        trial["production"] = measure * trial["gain"]

        self.add_period(["fixation", "ready"])
        self.add_period("measure", duration=measure, after="fixation")
        self.add_period("set", after="measure")
        self.add_period("production", duration=2 * trial["production"], after="set")

        self.add_ob(1, where="fixation")
        self.set_ob(0, "production", where="fixation")
        self.add_ob(1, "ready", where="ready")
        self.add_ob(1, "set", where="set")

        # set ground truth
        gt = np.zeros((int(2 * trial["production"] / self.dt),))
        gt[int(trial["production"] / self.dt)] = 1
        self.set_groundtruth(gt, "production")

        return trial

    def _step(self, action):
        trial = self.trial
        terminated = False
        truncated = False
        reward = 0
        ob = self.ob_now
        gt = self.gt_now
        new_trial = False
        if self.in_period("fixation") and action != 0:
            new_trial = self.abort
            reward = self.rewards["abort"]
        if self.in_period("production") and action == 1:
            new_trial = True  # terminate
            # time from end of measure:
            t_prod = self.t - self.end_t["measure"]
            eps = abs(t_prod - trial["production"])
            # actual production time
            eps_threshold = self.prod_margin * trial["production"] + 25
            if eps > eps_threshold:
                reward = self.rewards["fail"]
            else:
                reward = (1.0 - eps / eps_threshold) ** 1.5
                reward = max(reward, 0.1)
                reward *= self.rewards["correct"]
                self.performance = 1

        return ob, reward, terminated, truncated, {"new_trial": new_trial, "gt": gt}


class MotorTiming(ngym.TrialEnv):
    """Agents have to produce different time intervals using different effectors (actions).

    Args:
        prod_margin: controls the interval around the ground truth production
            time within which the agent receives proportional reward.
    """

    #  TODO: different actions not implemented
    metadata = {  # noqa: RUF012
        "paper_link": "https://www.nature.com/articles/s41593-017-0028-6",
        "paper_name": """Flexible timing by temporal scaling of
         cortical responses""",
        "tags": ["timing", "go-no-go", "supervised"],
    }

    def __init__(self, dt=80, rewards=None, timing=None, prod_margin=0.2) -> None:
        super().__init__(dt=dt)
        self.prod_margin = prod_margin
        self.production_ind = [0, 1]
        self.intervals = [800, 1500]

        # Rewards
        self.rewards = {"abort": -0.1, "correct": +1.0, "fail": 0.0}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            "fixation": 500,  # XXX: not specified
            "cue": lambda: self.rng.uniform(1000, 3000),
            "set": 50,
        }
        if timing:
            self.timing.update(timing)

        self.abort = False
        # set action and observation space
        self.action_space = spaces.Discrete(2)  # (fixate, go)
        # Fixation, Interval indicator x2, Set
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(4,),
            dtype=np.float32,
        )

    def _new_trial(self, **kwargs):
        trial = {"production_ind": self.rng.choice(self.production_ind)}
        trial.update(kwargs)

        trial["production"] = self.intervals[trial["production_ind"]]

        self.add_period(["fixation", "cue", "set"])
        self.add_period("production", duration=2 * trial["production"], after="set")

        self.set_ob([1, 0, 0, 0], "fixation")
        ob = self.view_ob("cue")
        ob[:, 0] = 1
        ob[:, trial["production_ind"] + 1] = 1
        ob = self.view_ob("set")
        ob[:, 0] = 1
        ob[:, trial["production_ind"] + 1] = 1
        ob[:, 3] = 1
        # set ground truth
        gt = np.zeros((int(2 * trial["production"] / self.dt),))
        gt[int(trial["production"] / self.dt)] = 1
        self.set_groundtruth(gt, "production")

        return trial

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
        trial = self.trial
        terminated = False
        truncated = False
        reward = 0
        ob = self.ob_now
        gt = self.gt_now
        new_trial = False
        if self.in_period("fixation") and action != 0:
            new_trial = self.abort
            reward = self.rewards["abort"]
        if self.in_period("production") and action == 1:
            new_trial = True  # terminate
            t_prod = self.t - self.end_t["set"]  # time from end of measure
            eps = abs(t_prod - trial["production"])
            # actual production time
            eps_threshold = self.prod_margin * trial["production"] + 25
            if eps > eps_threshold:
                reward = self.rewards["fail"]
            else:
                reward = (1.0 - eps / eps_threshold) ** 1.5
                reward = max(reward, 0.1)
                reward *= self.rewards["correct"]
                self.performance = 1

        return ob, reward, terminated, truncated, {"new_trial": new_trial, "gt": gt}


class OneTwoThreeGo(ngym.TrialEnv):
    """Agents reproduce time intervals based on two samples.

    Args:
        prod_margin: controls the interval around the ground truth production
                    time within which the agent receives proportional reward
    """

    metadata = {  # noqa: RUF012
        "paper_link": "https://www.nature.com/articles/s41593-019-0500-6",
        "paper_name": "Internal models of sensorimotor integration regulate cortical dynamics",
        "tags": ["timing", "go-no-go", "supervised"],
    }

    def __init__(self, dt=80, rewards=None, timing=None, prod_margin=0.2) -> None:
        super().__init__(dt=dt)

        self.prod_margin = prod_margin

        # Rewards
        self.rewards = {"abort": -0.1, "correct": +1.0, "fail": 0.0}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            "fixation": TruncExp(400, 100, 800),
            "target": TruncExp(1000, 500, 1500),
            "s1": 100,
            "interval1": (600, 700, 800, 900, 1000),
            "s2": 100,
            "interval2": 0,
            "s3": 100,
            "interval3": 0,
            "response": 1000,
        }
        if timing:
            self.timing.update(timing)

        self.abort = False
        # set action and observation space
        name = {"fixation": 0, "stimulus": 1, "target": 2}
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(3,),
            dtype=np.float32,
            name=name,
        )
        name = {"fixation": 0, "go": 1}
        self.action_space = spaces.Discrete(2, name=name)

    def _new_trial(self, **kwargs):
        interval = self.sample_time("interval1")
        trial = {
            "interval": interval,
        }
        trial.update(kwargs)

        self.add_period(["fixation", "target", "s1"])
        self.add_period("interval1", duration=interval, after="s1")
        self.add_period("s2", after="interval1")
        self.add_period("interval2", duration=interval, after="s2")
        self.add_period("s3", after="interval2")
        self.add_period("interval3", duration=interval, after="s3")
        self.add_period("response", after="interval3")

        self.add_ob(1, where="fixation")
        self.add_ob(1, ["s1", "s2", "s3"], where="stimulus")
        self.add_ob(1, where="target")
        self.set_ob(0, "fixation", where="target")

        # set ground truth
        self.set_groundtruth(1, period="response")

        return trial

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
        trial = self.trial
        terminated = False
        truncated = False
        reward = 0
        ob = self.ob_now
        gt = self.gt_now
        new_trial = False
        if self.in_period("interval3") or self.in_period("response"):
            if action == 1:
                new_trial = True  # terminate
                # time from end of measure:
                t_prod = self.t - self.end_t["s3"]
                eps = abs(t_prod - trial["interval"])  # actual production time
                eps_threshold = self.prod_margin * trial["interval"] + 25
                if eps > eps_threshold:
                    reward = self.rewards["fail"]
                else:
                    reward = (1.0 - eps / eps_threshold) ** 1.5
                    reward = max(reward, 0.1)
                    reward *= self.rewards["correct"]
                    self.performance = 1
        elif action != 0:
            new_trial = self.abort
            reward = self.rewards["abort"]

        return ob, reward, terminated, truncated, {"new_trial": new_trial, "gt": gt}
