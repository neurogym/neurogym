from typing import Any

import numpy as np

from neurogym.core import TrialEnv
from neurogym.utils import spaces


class AnnubesEnv(TrialEnv):
    # A reserved keyword for catch trials.
    catch_kwd: str = "catch"

    # The upper and lower bounds for the catch probability
    # used in the max_sequential satisfiability checks.
    prob_lbound: float = 1e-6
    prob_ubound: float = 1 - 1e-6

    def __init__(
        self,
        session: dict[str | tuple[str], float],
        catch_prob: float,
        intensities: dict[str, list[float]] | None = None,
        stim_time: int = 1000,
        max_sequential: dict[str, int] | int | None = None,
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
        """General class for the Annubes type of tasks.

        Args:
            session: Configuration dictionary for the types of trials that can appear during a session.
                The keys are modalities or combinations thereof, and the values are the probabilities of
                those appearing in a trial.
                For instance, `{"v": 0.25, "a": 0.75}` means that `a` would occur three times as often as `v`,
                which is the same as setting the session to `{"v": 1, "a": 3}`.
                Furthermore, the probability of a catch trial is given separately (cf. `catch_prob` below).
                For instance, if the catch probability is `0.5`, the probabilities will be used only
                in the remaining (`1 - catch_prob`) of the trials.
                NOTE:
                    - Probabilities are interpreted as being relative to each other.
                    - All keys in the session dictionary are converted into tuples for consistency.
                    - 'catch' is currently a reserved keyword that cannot be used as a key in the sesion dictionary.
                        This will change in the future when the definition of catch trials becomes part of the session.
            catch_prob: Probability of catch trials in the session. Must be between 0 and 1 (inclusive).
            intensities: A dictionary mapping each modality to a list of its possible intensity values.
                NOTE: If the modality is not present during a trial, its intensity is set to 0.
            stim_time: Duration of each stimulus in ms.
            max_sequential: Maximum number of sequential trials of the same stimulus type.
                By default, it applies only to the stimulus types defined in `session`, i.e.,
                it does not apply to catch trials. You can use `None` as a key in the `max_sequential`
                to set an explicit limit for sequential catch trials.
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
            iti: Inter-trial interval, or time window between sequential trials, in ms. Same format as `fix_time`.
            dt: Time step in ms.
            tau: Time constant in ms.
            output_behavior: List of possible intensity values of the behavioral output.
                Currently only the smallest and largest value of this list are used.
            noise_std: Standard deviation of the input noise.
            rewards: Dictionary of rewards for different outcomes. The keys are "abort", "correct", and "fail".
            random_seed: Seed for numpy's random number generator (rng).
                If an int is given, it will be used as the seed for `np.random.default_rng()`.

        Raises:
            ValueError: Raised if not all modalities have assigned intensities.
            ValueError: Raised if the conditions imposed by max_sequential are not satisfiable.

        """
        super().__init__(dt=dt)

        if not (0.0 <= catch_prob <= 1.0):
            msg = "The catch probability must be between 0 and 1, inclusive."
            raise ValueError(msg)
        self.catch_prob = catch_prob

        self.session = session
        self.modalities: set = set()
        self._check_session()
        self.prepare_session_and_modalities()
        # Check if the session is properly formatted and the probabilities are sane.

        # Ensure that each modality has an associated intensity.
        # The set of modalities is authoritative, so we can just compare
        # it with the set of keys in the `intensities` dictionary.
        if intensities is None:
            intensities = {modality: [0.8, 0.9, 1.0] for modality in self.modalities}
        self.intensities = intensities
        # Check if intensities have been provided for each modality.
        self._check_intensities()

        # Checks on the output behaviour and intensities
        if output_behavior is None:
            output_behavior = [0, 1]

        self.max_sequential = max_sequential
        self.prepare_max_sequential()
        # Check if the max_sequential constraints can be satisfied.
        self._check_max_sequential()

        self.stim_time = stim_time
        self.sequential_count = dict.fromkeys(self.max_sequential, 0)  # type: ignore[arg-type]
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
            self.rewards = {"abort": -0.1, "correct": 1.0, "fail": 0.0}
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
            **{trial: i for i, trial in enumerate(sorted(self.modalities), 2)},
        }
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(len(obs_space_name),), name=obs_space_name)
        # Set the name of each action value
        self.action_space = spaces.Discrete(
            n=len(self.output_behavior),
            name={"fixation": self.fix_intensity, "choice": self.output_behavior[1:]},
        )

    def _check_session(self):
        """Check if the session dictionary is valid.

        Args:
            session: A session dictionary.

        Raises:
            TypeError: Raised if the session is not a dictionary.
            ValueError: Raised if any of the probabilities are negative.
            ValueError: Raised if the probabilities effectively add up to 0.
        """
        # Run some common-sense checks on the session dictionary.
        if not isinstance(self.session, dict):
            msg = "The session must be a dictionary."
            raise TypeError(msg)

        if any(s < 0.0 for s in self.session.values()):
            msg = "Please ensure that all probabilities are non-negative."
            raise ValueError(msg)

        # Ensure that at least one probability is nonzero.
        if np.isclose(sum(self.session.values()), 0.0):
            msg = "Please ensure that at least one modality has a non-zero probability."
            raise ValueError(msg)

        if AnnubesEnv.catch_session_key in self.session or AnnubesEnv.catch_kwd in self.session:
            msg = (
                f"For now, '{AnnubesEnv.catch_kwd}' is a reserved keyword "
                "that cannot be used in the session dictionary. "
                "This warning will be removed in future releases when the catch trial "
                "definition becomes part of the session dictionary."
            )
            raise KeyError(msg)

    def _check_intensities(self):
        """Check that each modality has a corresponding intensity setting.

        Raises:
            ValueError: Raised if not all modalities have valid intensities.
        """
        if set(self.intensities) != self.modalities or any(
            not isinstance(self.intensities[mod], (list, tuple, float)) for mod in self.modalities
        ):
            msg = "Please ensure that all modalities have valid corresponding intensities."
            raise ValueError(msg)

    def _check_max_sequential(self):
        """Check the satisfiability of the conditions imposed by max_sequential.

        An example of an invalid max_sequential is if all of the following are true:
            - `catch_prob` is 0, so there are no catch trials.
            - The `session` dictionary contains one or more modalities that have a probability of 1,
                meaning that they should appear in every trial.
            - `max_sequential` limits how many times those modalities can be presented sequentially.

        Returns:
            An indication of whether the conditions imposed by max_sequential are satisfiable.
        """
        if AnnubesEnv.prob_lbound <= self.catch_prob <= AnnubesEnv.prob_ubound:
            # The probability of a catch trial is reasonable enough,
            # so we don't need to do any further checks.
            return

        if self.catch_prob > AnnubesEnv.prob_ubound:
            if sum(self.session.values()) > AnnubesEnv.prob_lbound:
                msg = (
                    "Invalid settings: the probability of catch trials is 1, "
                    "but the session contains stimuli with non-zero probabilities."
                )
                raise ValueError(msg)

            if self.max_sequential[AnnubesEnv.catch_kwd] is not None:
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

        # Compile a dictionary of available stimulus options.
        options: dict[tuple, float] = {}
        for stim, prob in self.session.items():
            if all(
                (
                    self.max_sequential[modality] is None # type ignore[index]
                    or self.sequential_count[modality] < self.max_sequential[modality]
                )
                for modality in stim
            ):
                options[stim] = (1 - self.catch_prob) * prob

        # If a catch trial is allowed, add the probability here
        if self.catch_prob > 0.0 and (
            self.max_sequential[AnnubesEnv.catch_kwd] is None
            or self.sequential_count[AnnubesEnv.catch_kwd] < self.max_sequential[AnnubesEnv.catch_kwd]
        ):
            options[AnnubesEnv.catch_session_key] = self.catch_prob

        # Select a stimulus from the available options.
        stimulus = self._select_stimulus(options)
        catch_trial = stimulus == AnnubesEnv.catch_session_key

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
            self.sequential_count[AnnubesEnv.catch_kwd] += 1

            stimulus = []
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
        options: dict,
    ) -> Any:
        """Select a sitmulus from the currently available options.

        Args:
            options: A dictionary of stimulus options mapped to their probabilities.

        Returns:
            A stimulus as a list containing more modalities or a catch trial.
        """
        # Select one of several mutually exclusive options.
        # First we need to normalise the probabilities since
        # they might not add up to 1.
        total = sum(options.values())
        flat_options = [(k, v / total) for k, v in options.items()]

        # Select an index at random
        idx = self._rng.choice(
            np.arange(len(flat_options)),
            p=[item[1] for item in flat_options],
        ).item()

        # Now pick the key from the array.
        # This is the stimulus that will be presented during the trial.
        return flat_options[idx][0]

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

    @staticmethod
    @property
    def catch_session_key():
        return AnnubesEnv.catch_session_key

    def prepare_session_and_modalities(self):
        """Validate the session keys, extract modalities and ensure that all probabilities are sane."""
        # Convert all keys into tuples for consistency.
        self.session = {tuple(sorted((k,) if isinstance(k, str) else k)): v for k, v in self.session.items()}

        # Validate the entries and extract the modalities.
        valid_session: dict = {}
        modalities = set()

        # This will be used to normalise the values.
        prob_sum = sum(self.session.values())

        for stim, prob in self.session.items():
            for modality in stim:
                if not isinstance(modality, str):
                    msg = f"Invalid modality type: {type(modality)}."
                    raise TypeError(msg)
                modalities.add(modality)

                # Add missing modalities with probability 0 to the session.
                # FIXME: This might not be necessary in general, but at present
                # the plotting function breaks if a modality is defined *only*
                # as part of a multi-modal stimulus and does not exist as a key
                # in the session dictionary.
                valid_session.setdefault((modality,), 0.0)

            # Sort the key for multimodal stimuli (facilitates testing).
            stim = tuple(sorted(stim))  # noqa: PLW2901

            valid_session[stim] = prob / prob_sum

        self.session = valid_session
        self.modalities = modalities

    def prepare_max_sequential(self):
        """Process and assign the respective max_sequential limits to each modality.

        NOTE: A value of 'None' means no limit.
        """
        if isinstance(self.max_sequential, dict):
            # Ensure that all the modalities are present in the dictionary.
            # The default value is None, which means no limit.
            for mod in self.modalities:
                self.max_sequential.setdefault(mod)
        elif self.max_sequential is None or isinstance(self.max_sequential, int):
            # If max_sequential is not a dictionary, use it as the default value
            # for a dictionary based on all modalities as keys.
            self.max_sequential = dict.fromkeys(self.modalities, self.max_sequential)
        else:
            msg = "'max_sequential' can only be a dictionary, an integer or None."
            raise TypeError(msg)

        # Set a limit on max_sequential for catch trials only if it is
        # explicitly set in the max_sequential dictionary from the outset.
        self.max_sequential.setdefault(AnnubesEnv.catch_kwd)
