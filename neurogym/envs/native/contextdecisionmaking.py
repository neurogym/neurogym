from typing import Any

import numpy as np

import neurogym as ngym
from neurogym import spaces
from neurogym.utils.ngym_random import TruncExp


class ContextDecisionMaking(ngym.TrialEnv):
    """Context-dependent decision-making task.

    The agent simultaneously receives stimulus inputs from two modalities (
    for example, a colored random dot motion pattern with color and motion
    modalities) and needs to make a perceptual decision based on one while
    ignoring the other. The task can operate in two modes:
    1. Implicit context: The relevant modality is fixed and must be learned (
       it is not explicitly signaled).
    2. Explicit context: The agent simultaneously receives stimulus inputs
       from two modalities, and the relevant modality is explicitly indicated
       by a context signal.
    Both modes use ring representation for encoding stimulus inputs and choices.

    Args:
        dt: int, timestep of the environment.
        use_expl_context: bool, if True, the context is explicit (signaled) and changes per trial.
        impl_context_modality: int, which fixed implicit modality to use if `use_expl_context`
            is False (0 or 1).
        dim_ring: int, number of choices.
        rewards: dict, rewards for correct, fail and abort responses.
        timing: dict, timing of the different events in the trial.
        sigma: float, standard deviation of the noise added to the inputs.
        abort: bool, if True, incorrect actions during fixation lead to trial abortion.
    """

    metadata = {  # noqa: RUF012
        "paper_link": "https://www.nature.com/articles/nature12742",
        "paper_name": """Context-dependent computation by recurrent
         dynamics in prefrontal cortex""",
        "tags": ["perceptual", "context dependent", "two-alternative", "supervised"],
    }

    def __init__(
        self,
        dt: int = 100,
        use_expl_context: bool = False,
        impl_context_modality: int = 0,
        dim_ring: int = 2,
        rewards: dict[str, float] | None = None,
        timing: dict[str, int | TruncExp] | None = None,
        sigma: float = 1.0,
        abort: bool = False,
    ) -> None:
        super().__init__(dt=dt)

        # Task parameters
        self.dt = dt
        self.use_expl_context = use_expl_context
        self.dim_ring = dim_ring
        self.sigma = sigma / np.sqrt(self.dt)

        # Trial conditions and spaces setup
        self.cohs: list[int] = [5, 15, 50]
        if use_expl_context:
            self.contexts: list[int] = [0, 1]
        else:
            self.contexts = [impl_context_modality]
        self._setup_spaces()

        # Rewards
        self.rewards: dict[str, float] = {"abort": -0.1, "correct": +1.0}
        if rewards:
            self.rewards.update(rewards)

        # Timing
        self.timing: dict[str, int | TruncExp] = {
            "fixation": 300,
            "stimulus": 750,
            "delay": TruncExp(600, 300, 3000),
            "decision": 100,
        }
        if timing:
            self.timing.update(timing)

        self.abort = abort

    def _setup_spaces(self):
        """Setup observation and action spaces using ring representation."""
        self.theta = np.linspace(0, 2 * np.pi, self.dim_ring + 1)[:-1]
        self.choices = np.arange(1, self.dim_ring + 1)

        # Base observation space with fixation and modalities
        obs_space_name = {
            "fixation": 0,
            **{f"stim{i + 1}_mod1": i + 1 for i in range(self.dim_ring)},
            **{f"stim{i + 1}_mod2": i + 1 + self.dim_ring for i in range(self.dim_ring)},
        }

        if self.use_expl_context:
            context_idx = 1 + 2 * self.dim_ring
            obs_space_name.update(
                {
                    "context1": context_idx,
                    "context2": context_idx + 1,
                },
            )

        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(len(obs_space_name),),
            dtype=np.float32,
            name=obs_space_name,
        )

        # Action space: fixation + choices
        action_space_name = {"fixation": 0, "choice": list(range(1, self.dim_ring + 1))}

        self.action_space = spaces.Discrete(
            n=1 + self.dim_ring,
            name=action_space_name,
        )

    def _new_trial(self, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        """New trial within the current context.

        Returns:
            dict: Trial information
        """
        # Trial parameters
        trial = {
            "ground_truth": self.rng.choice(self.choices),
            "other_choice": self.rng.choice(self.choices),
            "context": self.rng.choice(self.contexts),
            "coh_1": self.rng.choice(self.cohs),
            "coh_2": self.rng.choice(self.cohs),
        }
        trial.update(kwargs)

        choice_1, choice_2 = trial["ground_truth"], trial["other_choice"]
        if trial["context"] == 1:
            choice_2, choice_1 = choice_1, choice_2
        coh_1, coh_2 = trial["coh_1"], trial["coh_2"]

        # Add periods
        periods = ["fixation", "stimulus", "delay", "decision"]
        self.add_period(periods)

        # Set observations based on context type
        self.add_ob(1, where="fixation")

        self._set_observations(choice_1, choice_2, coh_1, coh_2)
        # Add context signals for explicit context
        if self.use_expl_context:
            if trial["context"] == 0:
                self.add_ob(1, where="context1")
            else:
                self.add_ob(1, where="context2")
        self.set_groundtruth(trial["ground_truth"], "decision")

        return trial

    def _set_observations(self, choice_1, choice_2, coh_1, coh_2):
        """Set observations for ring representation."""
        stim_theta_1 = self.theta[choice_1 - 1]
        stim_theta_2 = self.theta[choice_2 - 1]

        stim_mod1 = np.cos(self.theta - stim_theta_1) * (coh_1 / 200) + 0.5
        stim_mod2 = np.cos(self.theta - stim_theta_2) * (coh_2 / 200) + 0.5

        # Add observations for each index
        for i in range(self.dim_ring):
            self.add_ob(stim_mod1[i], "stimulus", where=f"stim{i + 1}_mod1")
            self.add_ob(stim_mod2[i], "stimulus", where=f"stim{i + 1}_mod2")
            self.add_randn(
                0,
                self.sigma,
                period="stimulus",
                where=f"stim{i + 1}_mod1",
            )
            self.add_randn(
                0,
                self.sigma,
                period="stimulus",
                where=f"stim{i + 1}_mod2",
            )

        self.set_ob(0, "decision")

    def _step(self, action: int) -> tuple:  # type: ignore[override]
        """Execute one timestep within the environment."""
        ob = self.ob_now
        gt = self.gt_now

        new_trial = False
        terminated = False
        truncated = False
        reward = 0.0

        if self.in_period("fixation") and action != 0:
            new_trial = self.abort
            reward = self.rewards["abort"]
        elif self.in_period("decision") and action != 0:
            new_trial = True
            if action == gt:
                reward = self.rewards["correct"]
                self.performance = 1

        return ob, reward, terminated, truncated, {"new_trial": new_trial, "gt": gt}
