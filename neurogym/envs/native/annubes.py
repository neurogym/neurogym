from typing import Any

import numpy as np

from neurogym.core import TrialEnv
from neurogym.utils import spaces


class AnnubesEnv(TrialEnv):
    """General class for the Annubes type of tasks.

    TODO: Move everything below to __init__.

    Args:
        session: Configuration dictionary for the types of trials that can appear during a session.
            The keys are modalities or combinations thereof, and the values are the probabilities of
            those appearing in a trial.
            NOTE: Probabilities are interpreted as being relative to each other.
            For instance, `{"v": 0.25, "a": 0.75}` means that `a` would occur three times as often as `v`,
            which is the same as setting the session to `{"v": 1, "a": 3}`.
            Furthermore, the probability of a catch trial is given separately (cf. `catch_prob` below).
            For instance, if the catch probability is `0.5`, the probabilities will be used only
            in the remaining (`1 - catch_prob`) of the trials.
            If the session is `None` or an empty dictionary, it defaults to `{"v": 0.5, "a": 0.5}`.
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
    """

    @staticmethod
    def _check_max_sequential_conflict(
        session: dict,
        modalities: set,
        catch_prob: float,
        max_sequential: dict | int | None,
        prepare: bool = False,
    ) -> bool:
        """Check the satisfiability of the the max_sequential condition.

        An example of a scenario where that could be a problem is if the catch probability
        is 0 and one or more modalities are supposed to appear 100% of the time,
        but at the same time max_sequential imposes a limit on how many times
        that stimulus can be presented.

        Args:
            session: A session dictionary with stimulus types.
            modalities: A tuple of modalities extracted from the session.
            catch_prob: The catch probability.
            max_sequential: An optional dictionary or integer specifying the
                number of consecutive stimulus presentations.
            prepare: Prepare the max_sequential dictionary first.
                This can be useful if the caller is a test function.

        Raises:
            ValueError: Raised if the max_sequential condition is not satisfiable.

        Returns:
            An indication of whether the max_sequential condition can be satisfied.
        """
        if prepare:
            # Get the properly formatted max_sequential dictionary.
            max_sequential = AnnubesEnv.prepare_max_sequential(
                session, modalities, catch_prob, max_sequential, check=False
            )

        rtol = float(10 * np.finfo(float).eps)
        atol = float(10 * np.finfo(float).eps)

        if not np.isclose(catch_prob, 0.0, rtol, atol):
            return False

        mod_probs = {mod: 0.0 for mod in modalities}
        for stim, prob in session.items():
            if not isinstance(stim, tuple):
                stim = (stim,)  # noqa: PLW2901
            for modality in stim:
                mod_probs[modality] += prob

        return any(max_sequential[k] is not None and np.isclose(v, 1.0, rtol, atol) for k, v in mod_probs.items())  # type: ignore[index]

    @staticmethod
    def prepare_session(
        session: dict[str | tuple[str, ...], float] | None,
    ) -> tuple[dict[str | tuple[str, ...], float], set[str]]:
        """Validate the session keys, extract modalities and nsure that all probabilities are sane.

        NOTE: This is a static method because it's also used by the AnnubesEnv tests.

        Args:
            session: The requested modalities (and combinations thereof) with their probabilities.

        Raises:
            AttributeError: Raised if the session is not a dictionary.
            AttributeError: Raised if the session dictionary is empty.
            ValueError: Raised if there are negative probabilities.
            ValueError: Raised if all probabilities specified in the session are 0.

        Returns:
            The validated session dictionary and the set of elementary modalities.
        """
        # Run some common sense checks on the session dictionary.
        if not isinstance(session, dict):
            msg = "The session must be a dictionary."
            raise TypeError(msg)

        if any(s < 0.0 for s in session.values()):
            msg = "Please ensure that all probabilities are non-negative."
            raise ValueError(msg)

        # Sum of all the probabilities in the session dictionary.
        # This will be used to normalise the values.
        probability_sum = sum(session.values())

        # Ensure that at least one probability is nonzero.
        if probability_sum < np.finfo(float).eps:
            msg = "Please ensure that at least one modality has a non-zero probability."
            raise ValueError(msg)

        # Validate the entries and extract the modalities.
        valid_session: dict = {}
        modalities = set()
        for stim, prob in session.items():
            if stim is None or isinstance(stim, str):
                modalities.add(stim)

            elif isinstance(stim, tuple):
                for element in stim:
                    if not (element is None or isinstance(element, str)):
                        msg = f"Invalid modality: {element}."
                        raise TypeError(msg)
                    modalities.add(element)

                    # Add missing modalities with probability 0 to the session.
                    # TODO: This might not be necessary in general, but at present
                    # the plotting function breaks if a modality is defined *only*
                    # as part of a multi-modal stimulus and does not exist as a key
                    # in the session dictionary.
                    valid_session.setdefault(element, 0.0)

                # Sort the key for multimodal stimuli (facilitates testing).
                stim = tuple(sorted(stim))  # noqa: PLW2901

            else:
                msg = f"Invalid session entry: {stim}."
                raise TypeError(msg)

            valid_session[stim] = prob / probability_sum

        return (valid_session, modalities)

    @staticmethod
    def prepare_max_sequential(
        session: dict,
        modalities: set,
        catch_prob: float,
        max_sequential: dict | int | None,
        check: bool = True,
    ) -> dict[str | None, int | None]:
        """Process and assign the respective max_sequential limits to each modality.

        Args:
            session: A session dictionary.
            modalities: A list of modalities extracted from the session dictionary.
            catch_prob: The catch probability.
            max_sequential: An optional dictionary or integer specifying the
                number of consecutive stimulus presentations.
            check: Check if the constraints imposed by max_sequential are actually feasible.

        Raises:
            ValueError: Raised if the max_sequential condition is not satisfiable.

        Returns:
            A dictionary specifying the number of consecutive stimulus presentations.
        """
        # Now populate max_sequential, including for catch trials ('None' key)
        default_max_sequential = max_sequential if isinstance(max_sequential, int) else None
        _max_sequential: dict = {}
        for modality in modalities:
            _max_sequential.setdefault(modality, default_max_sequential)
        if isinstance(max_sequential, dict):
            # Only set the max_sequential for catch trials if it is
            # explicitly set in the max_sequential dictionary.
            _max_sequential[None] = max_sequential.get(None, default_max_sequential)
        else:
            _max_sequential[None] = None
        max_sequential = _max_sequential

        if check and AnnubesEnv._check_max_sequential_conflict(session, modalities, catch_prob, max_sequential):
            msg = "Invalid settings: max_sequential imposes a limit on a modality that should appear in every trial."
            raise ValueError(msg)

        return max_sequential

    def __init__(
        self,
        session: dict[str | tuple[str, ...], float],
        catch_prob: float,
        intensities: dict[str, list[float]] | None = None,
        stim_time: int = 1000,
        max_sequential: dict[str | None, int | None] | int | None = None,
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
        # Validate the session and extract individual modalities.
        (session, modalities) = AnnubesEnv.prepare_session(session)

        # Ensure that each modality has an associated intensity.
        # The set of modalities is authoritative, so we can just compare
        # it with the set of keys in the `intensities` dictionary.
        if intensities is None:
            intensities = {modality: [0.8, 0.9, 1.0] for modality in modalities}
        if set(intensities) != set(modalities):
            msg = "Please ensure that all modalities have corresponding intensities."
            raise ValueError(msg)

        # Checks on the output behaviour and intensities
        if output_behavior is None:
            output_behavior = [0, 1]

        # Ensure that the max_sequential dictionary is
        # populated with values that make sense.
        max_sequential = AnnubesEnv.prepare_max_sequential(session, modalities, catch_prob, max_sequential)

        super().__init__(dt=dt)
        self.session = session
        self.modalities = modalities
        self.intensities = intensities
        self.stim_time = stim_time
        self.catch_prob = catch_prob
        self.max_sequential = max_sequential
        self.sequential_count = {mod: 0 for mod in max_sequential}
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
            **{trial: i for i, trial in enumerate(sorted(modalities), 2)},
        }
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(len(obs_space_name),), name=obs_space_name)
        # Set the name of each action value
        self.action_space = spaces.Discrete(
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

        # Compile a dictionary of available stimulus options.
        options: dict[str | tuple[str, ...] | None, float] = {}
        for stim, prob in self.session.items():
            if not isinstance(stim, tuple):
                stim = (stim,)  # noqa: PLW2901

            if all(
                (
                    self.max_sequential[modality] is None
                    or self.sequential_count[modality] < self.max_sequential[modality]  # type: ignore[operator]
                )
                for modality in stim
            ):
                options[stim] = (1 - self.catch_prob) * prob

        # If a catch trial is allowed, add the probability here
        if self.catch_prob > 0.0 and (
            self.max_sequential[None] is None or self.sequential_count[None] < self.max_sequential[None]  # type: ignore[operator]
        ):
            options[None] = self.catch_prob

        # Select a stimulus from the available options.
        modalities = self._select_modalities(options)
        catch_trial = modalities is None

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
            self.sequential_count[None] += 1

            modalities = []
            intensities: dict[str, float] = {}

        else:
            # Update the sequential count for the modalities that were selected.
            for modality in self.sequential_count:
                if modality in modalities:
                    self.sequential_count[modality] += 1
                else:
                    self.sequential_count[modality] = 0

            # Now we choose the values for the modalities
            intensities = {
                modality: self._rng.choice(self.intensities[modality], 1, False)[0] for modality in modalities
            }

            # Set the GT
            for modality in self.modalities:
                if modality in modalities:
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
            "stim_types": modalities,  # TODO: Rename the dictionary key to "modalities".
            "stim_values": intensities,  # TODO: Rename the dictionary key to "intensities".
            "sequential_count": self.sequential_count,
        }

        return self.trial

    def _select_modalities(
        self,
        options: dict,
    ) -> Any:
        """Pick sitmulus types and the corrsponding intensities.

        Args:
            options: The session options to choose from.

        Returns:
            A list of stimulus types.
        """
        # Select one of several mutually exclusive options.
        # Normalise first since the probabilities might not add up to 1.
        total = sum(options.values())
        flat_options = [(k, v / total) for k, v in options.items()]

        # Select an index at random
        idx = self._rng.choice(
            np.arange(len(flat_options)),
            1,
            False,
            p=[item[1] for item in flat_options],
        )[0]

        # Now pick the key from the array.
        # This is the final set of mocalities to be presented during the trial.
        modalities = flat_options[idx][0]

        if isinstance(modalities, str):
            modalities = [modalities]

        return modalities

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
