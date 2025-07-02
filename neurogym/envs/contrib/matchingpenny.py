import numpy as np

from neurogym.core import TrialEnv
from neurogym.utils import spaces
from neurogym.utils.logging import logger


class MatchingPenny(TrialEnv):
    """Matching penny task.

    The agent is rewarded when it selects the same target as the computer.
    opponent_type: Type of opponent. (def: 'mean_action', str)

    Args:
        learning_rate: learning rate in the mean_action opponent
    """

    metadata = {  # noqa: RUF012
        "paper_link": "https://www.nature.com/articles/nn1209",
        "paper_name": """Prefrontal cortex and decision making in a
         mixed-strategy game""",
        "tags": ["two-alternative"],
    }

    def __init__(
        self,
        dt=100,
        rewards=None,
        timing=None,
        opponent_type="mean_action",
        learning_rate=0.2,
    ) -> None:
        super().__init__(dt=dt)
        if timing is not None:
            logger.warning("Matching-Penny task does not require timing variable.")
        # TODO: remain to be carefully tested
        # Opponent Type
        self.opponent_type = opponent_type

        # Rewards
        self.rewards = {"correct": +1.0, "fail": 0.0}
        if rewards:
            self.rewards.update(rewards)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(2,),
            dtype=np.float32,
        )
        self.prev_opp_action = int(self.rng.rand() > 0.5)
        if self.opponent_type == "mean_action":
            self.mean_action = 0
            self.lr = learning_rate

    def _new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Trial (trials are one step long)
        # ---------------------------------------------------------------------
        # TODO: Add more types of opponents
        # determine the transitions
        if self.opponent_type == "random":
            opponent_action = int(self.rng.rand() > 0.5)
        elif self.opponent_type == "mean_action":
            opponent_action = 1 * (not np.round(self.mean_action))
        else:
            ot = self.opponent_type
            msg = f"Unknown opponent type {ot}."
            raise ValueError(msg)

        trial = {"opponent_action": opponent_action}
        self.ob = np.zeros((1, self.observation_space.shape[0]))
        self.ob[0, self.prev_opp_action] = 1
        self.prev_opp_action = trial["opponent_action"]
        self.gt = np.array([opponent_action])

        return trial

    def _step(self, action):
        trial = self.trial
        terminated = False
        truncated = False
        obs = self.ob[0]
        if self.opponent_type == "mean_action":
            self.mean_action += self.lr * (action - self.mean_action)
        if action == trial["opponent_action"]:
            reward = self.rewards["correct"]
            self.performance = 1
        else:
            reward = self.rewards["fail"]

        info = {"new_trial": True, "gt": self.gt}
        return obs, reward, terminated, truncated, info
