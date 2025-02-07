import numpy as np

import neurogym as ngym
from neurogym import spaces
from neurogym.utils.ngym_random import TruncExp


class UnifiedContextDecisionMaking(ngym.TrialEnv):
    """Context-dependent decision-making task.

    The agent simultaneously receives stimulus inputs from two modalities (
    for example, a colored random dot motion pattern with color and motion
    modalities) and needs to make a perceptual decision based on one while
    ignoring the other. The task can operate in two modes:
    1. Implicit context: The relevant modality is fixed and must be learned (
       it is not explicitly signaled). Uses ring representation.
    2. Explicit context: The agent simultaneously receives stimulus inputs
       from two modalities, and the relevant modality is explicitely indicated
       by a context signal. Uses direct value representation.

    Args:
        dt: int, timestep of the environment
        explicit_context: bool, if True, context changes per trial and is signaled.
            Also determines the representation type (direct values if True, ring if False)
        fixed_context: int, which context to use if `explicit_context` is False (0 or 1)
        dim_ring: int, number of choices when using ring representation (only used if `explicit_context` is False)
        rewards: dict, rewards for correct, fail and abort responses
        timing: dict, timing of the different events in the trial
        sigma: float, standard deviation of the noise added to the inputs
    """

    metadata = {  # noqa: RUF012
        "paper_link": "https://www.nature.com/articles/nature12742",
        "paper_name": """Context-dependent computation by recurrent
         dynamics in prefrontal cortex""",
        "tags": ["perceptual", "context dependent", "two-alternative", "supervised"],
    }

    def __init__(
        self,
        dt=100,
        explicit_context=False,
        fixed_context=0,
        dim_ring=2,
        rewards=None,
        timing=None,
        sigma=1.0,
    ) -> None:
        super().__init__(dt=dt)

        # Task parameters
        self.explicit_context = explicit_context
        self.context = fixed_context if not explicit_context else None  # Store fixed context for implicit case
        self.dim_ring = dim_ring
        self.sigma = sigma / np.sqrt(self.dt)

        # Trial conditions
        self.cohs = [5, 15, 50]
        if explicit_context:
            self.contexts = [0, 1]
        else:
            self.contexts = [self.context]

        # Rewards
        self.rewards = {"abort": -0.1, "correct": +1.0}
        if rewards:
            self.rewards.update(rewards)

        # Timing
        self.timing = {
            "fixation": 300,
            "stimulus": 750,
            "delay": TruncExp(600, 300, 3000),
            "decision": 100,
        }
        if timing:
            self.timing.update(timing)

        self.abort = False

        # Setup spaces based on context type
        if explicit_context:
            self._setup_direct_spaces()
        else:
            self._setup_ring_spaces()

    def _setup_ring_spaces(self):
        """Setup observation and action spaces for ring representation."""
        self.theta = np.linspace(0, 2 * np.pi, self.dim_ring + 1)[:-1]
        self.choices = np.arange(self.dim_ring)

        # Observation space: fixation + 2 modalities of ring inputs
        name = {
            "fixation": 0,
            "stimulus_mod1": range(1, self.dim_ring + 1),
            "stimulus_mod2": range(self.dim_ring + 1, 2 * self.dim_ring + 1),
        }

        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(1 + 2 * self.dim_ring,),
            dtype=np.float32,
            name=name,
        )

        # Action space: fixation + choices
        name = {"fixation": 0, "choice": range(1, self.dim_ring + 1)}
        self.action_space = spaces.Discrete(1 + self.dim_ring, name=name)

    def _setup_direct_spaces(self):
        """Setup observation and action spaces for direct value representation (explicit context)."""
        names = [
            "fixation",
            "stim1_mod1",
            "stim2_mod1",
            "stim1_mod2",
            "stim2_mod2",
            "context1",
            "context2",
        ]
        name = {name: i for i, name in enumerate(names)}
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(7,),
            dtype=np.float32,
            name=name,
        )

        name = {"fixation": 0, "choice1": 1, "choice2": 2}
        self.action_space = spaces.Discrete(3, name=name)
        self.choices = [1, 2]  # Fixed choices for direct representation

    def _new_trial(self, **kwargs):
        # Trial parameters
        trial = {
            "ground_truth": self.rng.choice(self.choices),
            "other_choice": self.rng.choice(self.choices),
            "context": self.rng.choice(self.contexts),
            "coh_0": self.rng.choice(self.cohs),
            "coh_1": self.rng.choice(self.cohs),
        }
        trial.update(kwargs)

        choice_0, choice_1 = trial["ground_truth"], trial["other_choice"]
        if trial["context"] == 1:
            choice_1, choice_0 = choice_0, choice_1
        coh_0, coh_1 = trial["coh_0"], trial["coh_1"]

        # Add periods
        periods = ["fixation", "stimulus", "delay", "decision"]
        self.add_period(periods)

        # Set observations based on context type
        self.add_ob(1, where="fixation")

        if self.explicit_context:
            self._set_direct_observations(choice_0, choice_1, coh_0, coh_1)
            # Add context signals
            if trial["context"] == 0:
                self.add_ob(1, where="context1")
            else:
                self.add_ob(1, where="context2")
            self.set_groundtruth(trial["ground_truth"], "decision")  # Direct value case
        else:
            self._set_ring_observations(choice_0, choice_1, coh_0, coh_1)
            self.set_groundtruth(trial["ground_truth"], "decision", where="choice")  # Ring case

        return trial

    def _set_ring_observations(self, choice_0, choice_1, coh_0, coh_1):
        """Set observations for ring representation (implicit context)."""
        stim_theta_0 = self.theta[choice_0]
        stim_theta_1 = self.theta[choice_1]

        stim = np.cos(self.theta - stim_theta_0) * (coh_0 / 200) + 0.5
        self.add_ob(stim, "stimulus", where="stimulus_mod1")
        stim = np.cos(self.theta - stim_theta_1) * (coh_1 / 200) + 0.5
        self.add_ob(stim, "stimulus", where="stimulus_mod2")

        self.add_randn(0, self.sigma, period="stimulus", where="stimulus_mod1")
        self.add_randn(0, self.sigma, period="stimulus", where="stimulus_mod2")
        self.set_ob(0, "decision")

    def _set_direct_observations(self, choice_0, choice_1, coh_0, coh_1):
        """Set observations for direct value representation (explicit context)."""
        signed_coh_0 = coh_0 if choice_0 == 1 else -coh_0
        signed_coh_1 = coh_1 if choice_1 == 1 else -coh_1

        self.add_ob((1 + signed_coh_0 / 100) / 2, period="stimulus", where="stim1_mod1")
        self.add_ob((1 - signed_coh_0 / 100) / 2, period="stimulus", where="stim2_mod1")
        self.add_ob((1 + signed_coh_1 / 100) / 2, period="stimulus", where="stim1_mod2")
        self.add_ob((1 - signed_coh_1 / 100) / 2, period="stimulus", where="stim2_mod2")

        self.add_randn(0, self.sigma, "stimulus")
        self.set_ob(0, "decision")

    def _step(self, action):
        """Execute one step in the environment."""
        ob = self.ob_now
        gt = self.gt_now

        new_trial = False
        terminated = False
        truncated = False
        reward = 0

        if self.in_period("fixation"):
            if action != 0:  # Broke fixation
                new_trial = self.abort
                reward = self.rewards["abort"]
        elif self.in_period("decision") and action != 0:
            new_trial = True
            if action == gt:
                reward = self.rewards["correct"]
                self.performance = 1

        return ob, reward, terminated, truncated, {"new_trial": new_trial, "gt": gt}
