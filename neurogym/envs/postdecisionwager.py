import numpy as np

import neurogym as ngym
from neurogym import spaces
from neurogym.utils.ngym_random import TruncExp


class PostDecisionWager(ngym.TrialEnv):
    """Post-decision wagering task assessing confidence.

    The agent first performs a perceptual discrimination task (see for more
    details the PerceptualDecisionMaking task). On a random half of the
    trials, the agent is given the option to abort the sensory
    discrimination and to choose instead a sure-bet option that guarantees a
    small reward. Therefore, the agent is encouraged to choose the sure-bet
    option when it is uncertain about its perceptual decision.
    """

    metadata = {  # noqa: RUF012
        "paper_link": "https://science.sciencemag.org/content/324/5928/759.long",
        "paper_name": """Representation of Confidence Associated with a
         Decision by Neurons in the Parietal Cortex""",
        "tags": ["perceptual", "delayed response", "confidence"],
    }

    def __init__(self, dt=100, rewards=None, timing=None, dim_ring=2, sigma=1.0) -> None:
        super().__init__(dt=dt)

        self.wagers = [True, False]
        self.theta = np.linspace(0, 2 * np.pi, dim_ring + 1)[:-1]
        self.choices = np.arange(dim_ring)
        self.cohs = [0, 3.2, 6.4, 12.8, 25.6, 51.2]
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {"abort": -0.1, "correct": +1.0, "fail": 0.0}
        if rewards:
            self.rewards.update(rewards)
        self.rewards["sure"] = 0.7 * self.rewards["correct"]

        self.timing = {
            "fixation": 100,
            # 'target':  0,  # noqa: ERA001
            "stimulus": TruncExp(180, 100, 900),
            "delay": TruncExp(1350, 1200, 1800),
            "pre_sure": lambda: self.rng.uniform(500, 750),
            "decision": 100,
        }
        if timing:
            self.timing.update(timing)

        self.abort = False

        # set action and observation space
        name = {"fixation": 0, "stimulus": [1, 2], "sure": 3}
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(4,),
            dtype=np.float32,
            name=name,
        )
        name = {"fixation": 0, "choice": [1, 2], "sure": 3}
        self.action_space = spaces.Discrete(4, name=name)

    # Input scaling
    @staticmethod
    def scale(coh):
        return (1 + coh / 100) / 2

    def _new_trial(self, **kwargs):
        # Trial info
        trial = {
            "wager": self.rng.choice(self.wagers),
            "ground_truth": self.rng.choice(self.choices),
            "coh": self.rng.choice(self.cohs),
        }
        trial.update(kwargs)
        coh = trial["coh"]
        ground_truth = trial["ground_truth"]
        stim_theta = self.theta[ground_truth]

        # Periods
        periods = ["fixation", "stimulus", "delay"]
        self.add_period(periods)
        if trial["wager"]:
            self.add_period("pre_sure", after="stimulus")
        self.add_period("decision", after="delay")

        # Observations
        self.add_ob(1, ["fixation", "stimulus", "delay"], where="fixation")
        stim = np.cos(self.theta - stim_theta) * (coh / 200) + 0.5
        self.add_ob(stim, "stimulus", where="stimulus")
        self.add_randn(0, self.sigma, "stimulus")
        if trial["wager"]:
            self.add_ob(1, ["delay", "decision"], where="sure")
            self.set_ob(0, "pre_sure", where="sure")

        # Ground truth
        self.set_groundtruth(ground_truth, period="decision", where="choice")

        return trial

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
        trial = self.trial
        new_trial = False
        terminated = False
        truncated = False

        reward = 0
        gt = self.gt_now

        if self.in_period("fixation"):
            if action != 0:
                new_trial = self.abort
                reward = self.rewards["abort"]
        elif self.in_period("decision"):
            new_trial = True
            if action == 0:
                new_trial = False
            elif action == 3:  # sure option
                if trial["wager"]:
                    reward = self.rewards["sure"]
                    norm_rew = (reward - self.rewards["fail"]) / (self.rewards["correct"] - self.rewards["fail"])
                    self.performance = norm_rew
                else:
                    reward = self.rewards["abort"]
            elif action == trial["ground_truth"]:
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
