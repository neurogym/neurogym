from typing import Any

import numpy as np

import neurogym as ngym
from neurogym import TrialEnv


class AnnubesEnv(TrialEnv):
    """General class for the Annubes type of tasks.

    The probabilities for different sensory modalities are specified using a dictionary used in the task.
    Depending on the value of the `exclusive` argument (cf. below), the stimuli can be presented sequentially
    or in parallel with a certain probability. Note that the probabilities for the different modalities are
    interpreted as being relative to each other, such that e.g. `{"v": 0.25, "a": 0.75}` is equivalent to
    `{"v": 1, "a": 3}`.

    Furthermore, note that the probability of catch trials is given separately (cf. `catch_prob` below).
    For instance, if the catch probability is `0.5`, the stimulus probabilities will be
    used only in the remaining (`1 - catch_prob`) of the trials.

    If the `exclusive` argument is set to `False`, the modalities are first chosen with _independent_
    draws for each modality using its corresponding probability. In this case, if none of the modalities
    are selected (which is always the case unless at least one of the modalities has a probability of `1`),
    then another draw is performed _as if the modalities are exclusive_, i.e., with normalised probability
    weights (in other words, as if the `exclusive` argument is set to `True`). This ensures that multiple
    modalities can appear in the trial with the correct joint probability while ensuring that `catch`
    trials also occur with the correct probability (`catch_prob`).

    Args:
        session: Configuration of the trials that can appear during a session.
            Defaults to {"v": 0.5, "a": 0.5}.
        stim_intensities: A dictionary of stimulus types mapped to possible intensity values of each stimulus
            that is actually presented to the agent with non-zero probability. Defaults to None.
        stim_time: Duration of each stimulus in ms.
            Defaults to 1000.
        catch_prob: Probability of catch trials in the session. Must be between 0 and 1 (inclusive).
            Defaults to 0.5.
        exclusive: This effectively switches between two trial modes:
                - True: Makes the modalities mutually exclusive, so they can be presented sequentially,
                but not together.
                - False: Depending on the probabilities provided for each modality, there could
                    be an overlap where multiple modalities are presented at the same time.
            Defaults to False.
        max_sequential: Maximum number of sequential trials of the same modality. It applies only to the modalities
            defined in `session`, i.e., it does not apply to catch trials.
            Defaults to None (no maximum).
        fix_intensity: Intensity of input signal during fixation.
            Defaults to 0.
        fix_time: Fixation time specification. Can be one of the following:
            - A number (int or float): Fixed duration in milliseconds.
            - A callable: Function that returns the duration when called.
            - A list of numbers: Random choice from the list.
            - A tuple specifying a distribution:
                - ("uniform", (min, max)): Uniform distribution between min and max.
                - ("choice", [options]): Random choice from the given options.
                - ("truncated_exponential", [parameters]): Truncated exponential distribution.
                - ("constant", value): Always returns the given value.
                - ("until", end_time): Sets duration to reach the specified end time.
            The final duration is rounded down to the nearest multiple of the simulation timestep (dt).
            Note that the duration of each input and output signal is increased by this time.
            Defaults to 500.
        iti: Inter-trial interval, or time window between sequential trials, in ms. Same format as `fix_time`.
            Defaults to 0.
        dt: Time step in ms.
            Defaults to 100.
        tau: Time constant in ms.
            Defaults to 100.
        output_behavior: List of possible intensity values of the behavioral output. Currently only the smallest and
            largest value of this list are used.
            Defaults to [0, 1].
        noise_std: Standard deviation of the input noise.
            Defaults to 0.01.
        rewards: Dictionary of rewards for different outcomes. The keys are "abort", "correct", and "fail".
            Defaults to {"abort": -0.1, "correct": +1.0, "fail": 0.0}.
        random_seed: Seed for numpy's random number generator (rng). If an int is given, it will be used as the seed
            for `np.random.default_rng()`.
            Defaults to None (i.e. the initial state itself is random).
    """

    def __init__(
        self,
        session: dict[str, int | float] | None = None,
        stim_intensities: dict[str, list[float]] | None = None,
        stim_time: int = 1000,
        catch_prob: float = 0.5,
        exclusive: bool = False,
        max_sequential: dict[str, int | None] | int | None = None,
        fix_intensity: float = 0,
        fix_time: Any = 500,
        iti: Any = 0,
        dt: int = 100,
        tau: int = 100,
        output_behavior: list[float] | None = None,
        noise_std: float = 0.01,
        rewards: dict[str, float] | None = None,
        random_seed: int | None = None,
    ):
        # A session essentially represents a specification for presenting
        # different sensory modalities and their respective probabilities.
        if session is None:
            session = {"v": 0.5, "a": 0.5}

        # Ensure that the probabilities are sane.
        if any(not (0.0 <= s <= 1.0) for s in session.values()):
            msg = "Please ensure that all probabilities are between 0 and 1, inclusive."
            raise ValueError(msg)

        # Normalise the probabilities if the 'exclusive' switch is on
        if exclusive:
            total = sum(session.values())
            if total < np.nextafter(0, 1):
                msg = "Please ensure that at least one modality has a non-zero probability."
                raise ValueError(msg)
            session = {k: v / total for k, v in session.items()}

        if output_behavior is None:
            output_behavior = [0, 1]
        if stim_intensities is None:
            stim_intensities = {k: [0.8, 0.9, 1.0] for k in session}

        # Create a dictionary for max_sequential if only an int is given
        if isinstance(max_sequential, int):
            max_sequential = dict.fromkeys(session, max_sequential)

        super().__init__(dt=dt)
        self.session = {k: v for k, v in session.items() if v > 0.0}
        self.stim_intensities = stim_intensities
        self.stim_time = stim_time
        self.catch_prob = catch_prob
        self.exclusive = exclusive
        self.max_sequential = max_sequential
        self.sequential_count = dict.fromkeys(self.session, 0)
        # Sequential occurrence checks for catch trials
        if max_sequential is not None and None in max_sequential:
            self.sequential_count[None] = 0
        self.fix_intensity = fix_intensity
        self.fix_time = fix_time
        self.iti = iti
        self.dt = dt
        self.tau = tau
        self.output_behavior = output_behavior
        self.noise_std = noise_std
        self.random_seed = random_seed
        alpha = dt / self.tau
        self.noise_factor = self.noise_std * np.sqrt(2 * alpha) / alpha
        # Set random state
        if random_seed is None:
            rng = np.random.default_rng(random_seed)
            self._random_seed = rng.integers(2**32)
        else:
            self._random_seed = random_seed
        self._rng = np.random.default_rng(self._random_seed)
        # Rewards
        if rewards is None:
            self.rewards = {"abort": -0.1, "correct": +1.0, "fail": 0.0}
        else:
            self.rewards = rewards
        self.timing = {
            "fixation": self.fix_time,
            "stimulus": self.stim_time,
            "iti": self.iti,
        }
        # Set the name of each input dimension
        obs_space_name = {
            "fixation": 0,
            "start": 1,
            **{trial: i for i, trial in enumerate(session, 2)},
        }
        self.observation_space = ngym.spaces.Box(low=0.0, high=1.0, shape=(len(obs_space_name),), name=obs_space_name)
        # Set the name of each action value
        self.action_space = ngym.spaces.Discrete(
            n=len(self.output_behavior),
            name={"fixation": self.fix_intensity, "choice": self.output_behavior[1:]},
        )

    def _new_trial(self, **kwargs: Any) -> dict:  # type: ignore[override]
        """Internal method to generate a new trial.

        Returns:
            A dictionary containing the information of the new trial.
        """
        # Setting time periods and their order for this trial
        self.add_period(["fixation", "stimulus", "iti"])

        # Adding fixation and start signal values
        self.add_ob(self.fix_intensity, "fixation", where="fixation")
        self.add_ob(1, "stimulus", where="start")

        # First, check if we have any available modalities
        available_modalities = {}
        for k, v in self.session.items():
            if (
                self.max_sequential is None
                or k not in self.max_sequential
                or self.sequential_count[k] < self.max_sequential[k]
            ):
                available_modalities[k] = v

        # Total probability weights
        total = sum(available_modalities.values())

        # Normalise the probability weights
        available_modalities = {k: v / total for k, v in available_modalities.items()}

        # Catch trial decision
        if (
            self.max_sequential is not None
            and None in self.max_sequential
            and self.sequential_count[None] == self.max_sequential[None]
        ):
            catch = False
        else:
            catch = len(available_modalities) == 0 or self._rng.random() < self.catch_prob
        stim_types = set()
        stim_values = {}

        if catch:
            self.set_groundtruth(0, period="fixation")
            self.set_groundtruth(0, period="stimulus")
            self.set_groundtruth(0, period="iti")

            # Reset the sequential count for all modalities
            self.sequential_count = {k: 0 if k is not None else v for k, v in self.sequential_count.items()}

        else:
            stim_types = self._pick_stim_types(available_modalities)
            self.sequential_count = {k: (v + 1 if k in stim_types else 0) for k, v in self.sequential_count.items()}

            # Reset the sequential count for catch trials
            if None in self.sequential_count:
                self.sequential_count[None] = 0

            stim_values = {k: self._rng.choice(self.stim_intensities[k], 1, False)[0] for k in stim_types}

            for mod in self.session:
                if mod in stim_types:
                    self.add_ob(stim_values[mod], "stimulus", where=mod)
                    self.add_randn(0, self.noise_factor, "stimulus", where=mod)
                self.set_groundtruth(0, period="fixation")
                self.set_groundtruth(1, period="stimulus")
                self.set_groundtruth(0, period="iti")

        stim_types_final = list(stim_types)

        self.trial = {
            "catch": catch,
            "stim_types": stim_types_final,
            "stim_values": [stim_values[k] for k in stim_types_final],
            "sequential_count": self.sequential_count,
        }

        return self.trial

    def _pick_stim_types(self, available: dict) -> set:
        if self.exclusive:
            # Mutually exclusive modalities. Pick *only* one modality.
            stim_types = set(
                self._rng.choice(
                    list(available.keys()),
                    1,
                    False,
                    p=list(available.values()),
                )[0],
            )

        else:
            # Overlap is permitted. Pick *at least* one modality.
            stim_types = {k for k, v in available.items() if self._rng.random() <= v}
            if len(stim_types) == 0:
                # Pick at least one modality from the available ones
                stim_types = set(
                    self._rng.choice(
                        list(available.keys()),
                        1,
                        False,
                        list(available.values()),
                    )[0],
                )

        return stim_types

    def _step(self, action: int) -> tuple:  # type: ignore[override]
        """Internal method to compute the environment's response to the agent's action.

        Args:
            action: The agent's action.

        Returns:
            A tuple containing the new observation, the reward, a boolean indicating whether the trial is
            terminated, a boolean indicating whether the trial is truncated, and a dictionary containing additional
            information.
        """
        new_trial = False
        terminated = False
        truncated = False
        reward = 0
        gt = self.gt_now

        if self.in_period("fixation") or self.in_period("iti"):
            if action != 0:
                reward += self.rewards["abort"]
        elif self.in_period("stimulus"):
            if action == gt:
                reward += self.rewards["correct"]
                self.performance = 1
            else:
                reward += self.rewards["fail"]

            # End trial when stimulus period is over
            # self.t represents the current time step within a trial
            # each step is self.dt ms
            # self.tmax is the maximum number of time steps within a trial
            # see self.add_period in TrialEnv for more details
            if self.t >= self.tmax - self.dt:
                new_trial = True

        info = {"new_trial": new_trial, "gt": gt}

        if new_trial:
            info["trial"] = self.trial
            self.trial = {}

        return self.ob_now, reward, terminated, truncated, info
