import numpy as np

import neurogym as ngym
from neurogym import spaces


class DelayComparison(ngym.TrialEnv):
    """Delayed comparison.

    The agent needs to compare the magnitude of two stimuli are separated by a
    delay period. The agent reports its decision of the stronger stimulus
    during the decision period.
    """

    metadata = {  # noqa: RUF012
        "paper_link": "https://www.jneurosci.org/content/30/28/9424",
        "paper_name": """Neuronal Population Coding of Parametric
        Working Memory""",
        "tags": ["perceptual", "working memory", "two-alternative", "supervised"],
    }

    def __init__(self, dt=100, vpairs=None, rewards=None, timing=None, sigma=1.0) -> None:
        super().__init__(dt=dt)

        # Pair of stimulus strengthes
        if vpairs is None:
            self.vpairs = [(18, 10), (22, 14), (26, 18), (30, 22), (34, 26)]
        else:
            self.vpairs = vpairs

        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {"abort": -0.1, "correct": +1.0, "fail": 0.0}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            "fixation": 500,
            "stimulus1": 500,
            "delay": 1000,
            "stimulus2": 500,
            "decision": 100,
        }
        if timing:
            self.timing.update(timing)

        self.abort = False

        # Input scaling
        self.vall = np.ravel(self.vpairs)
        self.vmin = np.min(self.vall)
        self.vmax = np.max(self.vall)

        # action and observation space
        name: dict[str, int | list] = {"fixation": 0, "stimulus": 1}
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(2,),
            dtype=np.float32,
            name=name,
        )
        name = {"fixation": 0, "choice": [1, 2]}
        self.action_space = spaces.Discrete(3, name=name)

        self.choices = [1, 2]

    def _new_trial(self, **kwargs):
        trial = {
            "ground_truth": self.rng.choice(self.choices),
            "vpair": self.vpairs[self.rng.choice(len(self.vpairs))],
        }
        trial.update(kwargs)

        v1, v2 = trial["vpair"]
        if trial["ground_truth"] == 2:
            v1, v2 = v2, v1
        trial["v1"] = v1
        trial["v2"] = v2

        # Periods
        periods = ["fixation", "stimulus1", "delay", "stimulus2", "decision"]
        self.add_period(periods)

        self.add_ob(1, where="fixation")
        self.add_ob(self.represent(v1), "stimulus1", where="stimulus")
        self.add_ob(self.represent(v2), "stimulus2", where="stimulus")
        self.set_ob(0, "decision")
        self.add_randn(0, self.sigma, ["stimulus1", "stimulus2"])

        self.set_groundtruth(trial["ground_truth"], "decision")

        return trial

    def represent(self, v):
        """Input representation of stimulus value."""
        # Scale to be between 0 and 1
        v_ = (v - self.vmin) / (self.vmax - self.vmin)
        # positive encoding, between 0.5 and 1
        return (1 + v_) / 2

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
        new_trial = False
        terminated = False
        truncated = False
        gt = self.gt_now
        ob = self.ob_now
        # rewards
        reward = 0
        if self.in_period("fixation"):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards["abort"]
        elif self.in_period("decision") and action != 0:
            new_trial = True
            if action == gt:
                reward = self.rewards["correct"]
                self.performance = 1
            else:
                reward = self.rewards["fail"]

        return ob, reward, terminated, truncated, {"new_trial": new_trial, "gt": gt}
