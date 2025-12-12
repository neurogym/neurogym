from copy import deepcopy
from typing import Any

import numpy as np

from neurogym.core import TrialEnv
from neurogym.utils import spaces

# A reserved keyword for catch trials.
CATCH_KEYWORD: str = "catch"
CATCH_SESSION_KEY: tuple[str] = (CATCH_KEYWORD,)

# The upper and lower bounds for the catch probability used in the max_sequential satisfiability checks.
MIN_CATCHPROB: float = 1e-6
MAX_CATCHPROB: float = 1 - 1e-6


class AnnubesEnv(TrialEnv):
    """Annubes task.

    Args:
        session: Configuration dictionary for the types of trials that can appear during a session.
            The keys are stimuli that can be presented during the trial and the values are the probabilities that that
            stimulus will appear in a trial. Each stimulus can be either a string (representing a single modality) or a
            tuple of strings (representing a combination of one or more modalities).
            Note that:
                - probabilities are interpreted as being relative to each other, such that e.g. `{"v": 0.25,
                "a": 0.75}` is identical to `{"v": 1, "a": 3}`.
                - the probability of a catch trial is given separately (cf. `catch_prob` below). For instance, if the
                catch probability is `0.5`, the probabilities will be used only in the remaining (`1 - catch_prob`) of
                the trials.
                - catch trials are defined by the reserved keyword `catch` in the session dictionary, which cannot be
                used as a user input.
        catch_prob: Probability of catch trials in the session. Must be between 0 and 1 (inclusive).
        intensities: Intensity values for each modality. Can be either:
            - A list of floats, which will be used for all modalities.
            - A dictionary mapping each modality to a list of its possible intensity values.
            - `None` (default), which will default to [0.8, 0.9, 1.0] for each modality.
        stim_time: Duration of each stimulus in ms. Defaults to 1000 ms.
        max_sequential: Maximum number of sequential trials of the same stimulus. Can be either:
            - an integer, which will apply to all stimuli defined in `session`, but NOT to catch trials.
            - a dictionary, which can specify an independent limit for each stimulus, including catch trials.
            - `None` (default), which means no limit on the number of sequential trials.
        fix_intensity: Intensity of input signal during fixation.
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
            Defaults to 500 ms.
        iti: Inter-trial interval, or time window between sequential trials, in ms. Same format as `fix_time`.
            Defaults to 0 ms, meaning no inter-trial interval.
        dt: Time step in ms. Defaults to 100 ms.
        tau: Time constant in ms. Defaults to 100 ms.
        output_behavior: List of possible intensity values of the behavioral output.
            Currently only the smallest and largest value of this list are used.
            Defaults to [0, 1].
        noise_std: Standard deviation of the input noise. Defaults to 0.01.
        rewards: Dictionary of rewards for different outcomes. The keys are "abort", "correct", and "fail".
            Defaults to: {"abort": -0.1, "correct": 1.0, "fail": 0.0}.
        random_seed: Optional seed for numpy's random number generator (rng).
        frozen_seed: If True, the random seed will be copied to the new environment instances when deepcopying. If
            False, the seed will be reset to a new reproducible quasi-random value.

    Raises:
        ValueError: Raised if the conditions imposed by max_sequential are not satisfiable.
    """

    metadata = {  # noqa: RUF012
        "paper_link": None,
        "paper_name": None,
        "tags": ["n-alternative", "perceptual", "supervised"],  # TODO: check these
    }

    def __init__(
        self,
        session: dict[str | tuple[str], float],
        catch_prob: float,
        intensities: list[float] | dict[str, list[float]] | None = None,
        stim_time: int = 1000,
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
        frozen_seed: bool = False,
    ):
        super().__init__(dt=dt)

        # Initialize the environment parameters.
        self.session, self.modalities = self._prepare_session_and_modalities(session)
        self.catch_prob = catch_prob

        if intensities is None:
            self.intensities = {modality: [0.8, 0.9, 1.0] for modality in self.modalities}
        elif isinstance(intensities, list):
            self.intensities = dict.fromkeys(self.modalities, intensities)
        elif isinstance(intensities, dict):
            self.intensities = intensities
        else:
            msg = "Intensities must be a list, a dictionary, or None."
            raise TypeError(msg)

        self.stim_time = stim_time
        self.max_sequential = self._prepare_max_sequential(max_sequential)
        self.fix_intensity = fix_intensity
        self.fix_time = fix_time
        self.iti = iti
        self.dt = dt
        self.tau = tau
        self.output_behavior = output_behavior or [0, 1]
        self.noise_std = noise_std
        self.rewards = rewards or {"abort": -0.1, "correct": 1.0, "fail": 0.0}
        if random_seed is None:
            rng = np.random.default_rng(random_seed)
            self._random_seed = rng.integers(2**32)
        else:
            self._random_seed = random_seed
        self._rng = np.random.default_rng(self._random_seed)
        self.frozen_seed = frozen_seed

        # Check if all input parameters are valid.
        self._check_inputs()

        # set derived environment parameters
        self.sequential_count = dict.fromkeys(self.max_sequential, 0)
        alpha = dt / self.tau
        self.noise_factor = self.noise_std * np.sqrt(2 * alpha) / alpha
        self.timing = {
            "fixation": self.fix_time,
            "stimulus": self.stim_time,
            "iti": self.iti,
        }
        # Set the name of each input dimension
        obs_space_name = {
            "fixation": 0,
            "start": 1,
            **{trial: i for i, trial in enumerate(sorted(self.modalities), 2)},
        }
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(len(obs_space_name),), name=obs_space_name)
        # Set the name of each action value
        self.action_space = spaces.Discrete(
            n=len(self.output_behavior),
            name={"fixation": self.fix_intensity, "choice": self.output_behavior[1:]},
        )

    def _check_inputs(self) -> None:
        """Check if inputs are is valid."""
        lowercase_modalities = [mod.lower() for mod in self.modalities]
        if CATCH_KEYWORD.lower() in lowercase_modalities:
            msg = f"Reserved keyword `{CATCH_KEYWORD}` cannot be given in the session dictionary."
            raise ValueError(msg)

        if any(s < 0.0 for s in self.session.values()):
            msg = "Please ensure that all probabilities are non-negative."
            raise ValueError(msg)

        if not (0.0 <= self.catch_prob <= 1.0):
            msg = "The catch probability must be between 0 and 1, inclusive."
            raise ValueError(msg)

        if set(self.intensities) != self.modalities or any(
            not isinstance(self.intensities[mod], (list, tuple, float)) for mod in self.modalities
        ):
            msg = "Please ensure that all modalities have valid corresponding intensities."
            raise ValueError(msg)

        if self.max_sequential:
            self._check_max_sequential()

    def _check_max_sequential(self) -> None:
        """Check the satisfiability of the conditions imposed by max_sequential.

        An example of an invalid max_sequential is if all of the following are true:
            - `catch_prob` is 0, so there are no catch trials.
            - The `session` dictionary contains one or more modalities that have a probability of 1,
                meaning that they should appear in every trial.
            - `max_sequential` limits how many times those modalities can be presented sequentially.

        Returns:
            An indication of whether the conditions imposed by max_sequential are satisfiable.
        """
        if MIN_CATCHPROB <= self.catch_prob <= MAX_CATCHPROB:
            # The probability of a catch trial is reasonable enough,
            # so we don't need to do any further checks.
            return

        if self.catch_prob > MAX_CATCHPROB:
            if sum(self.session.values()) > MIN_CATCHPROB:
                msg = (
                    "Invalid settings: the probability of catch trials is 1, "
                    "but the session contains stimuli with non-zero probabilities."
                )
                raise ValueError(msg)

            if self.max_sequential[CATCH_KEYWORD] is not None:
                msg = (
                    "Invalid settings: max_sequential imposes a limit on catch trials, "
                    "but the probability of a catch trial is too high."
                )
                raise ValueError(msg)

        # At this point we know that catch_prob is effectively 0,
        # so we also have to check the probabilities of all modalities.
        mod_probs = dict.fromkeys(self.modalities, 0.0)
        for stim, prob in self.session.items():
            for modality in stim:
                mod_probs[modality] += prob

        for mod, prob in mod_probs.items():
            if self.max_sequential[mod] is not None and np.isclose(prob, 1.0):
                msg = (
                    f"Invalid settings: max_sequential imposes a limit on modality '{mod}' "
                    "that should appear in every trial."
                )
                raise ValueError(msg)

        # Ensure that at least one probability is nonzero.
        if np.isclose(sum(self.session.values()), 0.0):
            msg = "Please ensure that at least one modality has a non-zero probability."
            raise ValueError(msg)

    def _new_trial(self, **kwargs) -> dict[str, Any]:  # type: ignore[override]
        """Internal method to generate a new trial.

        Returns:
            A dictionary containing the information of the new trial.
        """
        # Setting time periods and their order for this trial
        self.add_period(["fixation", "stimulus", "iti"])

        # Adding fixation and start signal values
        self.add_ob(self.fix_intensity, "fixation", where="fixation")
        self.add_ob(1, "stimulus", where="start")

        # Compile a dictionary of available stimulus options.
        options: dict[tuple[str, ...], float] = {}
        for stim, prob in self.session.items():
            if all(
                (
                    self.max_sequential[modality] is None
                    or self.sequential_count[modality]  # type: ignore[operator]
                    < self.max_sequential[modality]
                )
                for modality in stim
            ):
                options[stim] = (1 - self.catch_prob) * prob

        # If a catch trial is allowed, add the probability here
        if self.catch_prob > 0.0 and (
            self.max_sequential[CATCH_KEYWORD] is None
            or self.sequential_count[CATCH_KEYWORD]  # type: ignore[operator]
            < self.max_sequential[CATCH_KEYWORD]
        ):
            options[CATCH_SESSION_KEY] = self.catch_prob

        # Select a stimulus from the available options.
        stimulus = self._select_stimulus(options)
        catch_trial = stimulus == CATCH_SESSION_KEY

        if catch_trial:
            # Set all the GT values to 0.
            self.set_groundtruth(0, period="fixation")
            self.set_groundtruth(0, period="stimulus")
            self.set_groundtruth(0, period="iti")

            # Reset the sequential count for all modalities
            # since a catch trial breaks any sequences thereof.
            for modality in self.sequential_count:
                if modality is not None:
                    self.sequential_count[modality] = 0
            self.sequential_count[CATCH_KEYWORD] += 1

            stimulus = ()
            intensities: dict[str, float] = {}

        else:
            # Update the sequential count for the modalities that were selected.
            for modality in self.sequential_count:
                if modality in stimulus:
                    self.sequential_count[modality] += 1
                else:
                    self.sequential_count[modality] = 0

            # Now we choose the values for the modalities
            intensities = {modality: self._rng.choice(self.intensities[modality], 1, False)[0] for modality in stimulus}

            # Set the GT
            for modality in self.modalities:
                if modality in stimulus:
                    self.add_ob(intensities[modality], "stimulus", where=modality)
                    # Add noise to the
                    self.add_randn(0, self.noise_factor, "stimulus", where=modality)
                # REVIEW: Should this be outside the `for` loop?
                self.set_groundtruth(0, period="fixation")
                self.set_groundtruth(1, period="stimulus")
                self.set_groundtruth(0, period="iti")

        # Build the trial
        self.trial = {
            "catch": catch_trial,
            "stim_types": stimulus,  # TODO: Rename the dictionary key to "modalities".
            "stim_values": intensities,  # TODO: Rename the dictionary key to "intensities".
            "sequential_count": self.sequential_count,
        }

        return self.trial

    def _select_stimulus(
        self,
        options: dict[tuple[str, ...], float],
    ) -> tuple[str, ...]:
        """Select a stimulus for the next trial.

        Args:
            options: A dictionary of available stimulus options (taking max_sequential into account) mapped to their
            probabilities.

        Returns:
            A stimulus, represented as tuple of modalities.
        """
        # Normalise so probabilities add to 1.
        total = sum(options.values())
        normalized_options = [(k, v / total) for k, v in options.items()]  # list form of options: [(stim, prob)]

        # Select an index at random. We cannot directly use the keys of the dictionary because they are tuples,
        # and numpy's choice does not support tuples as keys very well.
        idx = self._rng.choice(
            len(normalized_options),
            p=[item[1] for item in normalized_options],
        )
        idx = int(idx)

        # Now pick the key from the array.
        # This is the stimulus that will be presented during the trial.
        return normalized_options[idx][0]

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

            # End the trial when the stimulus period is over.
            # self.t represents the current time step within a trial,
            # and each step is self.dt ms.
            # self.tmax is the maximum number of time steps within a trial.
            # See self.add_period in TrialEnv for more details.
            if self.t >= self.tmax - self.dt:
                new_trial = True

        info = {"new_trial": new_trial, "gt": gt}

        if new_trial:
            info["trial"] = self.trial
            self.trial = {}

        return self.ob_now, reward, terminated, truncated, info

    def _prepare_session_and_modalities(
        self,
        session: dict[str | tuple[str], float],
    ) -> tuple[dict[tuple[str, ...], float], set[str]]:
        """Return a properly formatted session dictionary and set of unique modalities."""
        # Convert all keys into tuples for consistency.
        if not isinstance(session, dict):
            msg = "The session must be a dictionary."
            raise TypeError(msg)

        temp_session = {tuple(sorted((k,) if isinstance(k, str) else k)): v for k, v in session.items()}

        # Validate the entries and extract the modalities.
        valid_session: dict[tuple[str, ...], float] = {}
        modalities = set()

        # This will be used to normalise the values.
        prob_sum = sum(temp_session.values())

        for stim, prob in temp_session.items():
            for modality in stim:
                if not isinstance(modality, str):
                    msg = f"Invalid modality type: {type(modality)}."
                    raise TypeError(msg)
                modalities.add(modality)

                # Add missing modalities with probability 0 to the session. This is required because the plotting
                # function breaks if a modality is defined *only* as part of a multi-modal stimulus and does not exist
                # as a key in the session dictionary.
                valid_session.setdefault((modality,), 0.0)

            # Sort the key for multimodal stimuli (facilitates testing).
            valid_stim = tuple(sorted(stim))

            try:
                valid_session[valid_stim] = prob / prob_sum
            except ZeroDivisionError as e:
                msg = (
                    "Normalization of probabilities failed. "
                    "Please ensure that all probabilities are non-negative and at least one is non-zero."
                )
                raise ValueError(msg) from e

        return valid_session, modalities

    def _prepare_max_sequential(self, max_sequential) -> dict[str, int | None]:
        """Process and assign the respective max_sequential limits to each modality.

        A value of 'None' means no limit.
        """
        valid_max_sequential: dict[str, int | None] = {}
        if isinstance(max_sequential, dict):
            # Ensure that all the modalities are present in the dictionary.
            # The default value is None, which means no limit.
            for mod in self.modalities:
                valid_max_sequential.setdefault(mod, None)
        elif max_sequential is None or isinstance(max_sequential, int):
            # If max_sequential is not a dictionary, use it as the default value
            # for a dictionary based on all modalities as keys.
            valid_max_sequential = dict.fromkeys(self.modalities, max_sequential)
        else:
            msg = "'max_sequential' can only be a dictionary, an integer or None."
            raise TypeError(msg)

        # Set a limit on max_sequential for catch trials only if it is
        # explicitly set in the max_sequential dictionary from the outset.
        valid_max_sequential.setdefault(CATCH_KEYWORD, None)

        return valid_max_sequential

    def __deepcopy__(self, memo):
        """Deep copy of AnnubesEnv, optionally with a new random state."""
        obj = self.__class__.__new__(self.__class__)
        memo[id(self)] = obj
        for k, v in self.__dict__.items():
            setattr(obj, k, deepcopy(v, memo))
        if not self.frozen_seed:
            rnd = self._rng.integers(2**32)
            self._rng = np.random.default_rng(rnd)
        return obj
