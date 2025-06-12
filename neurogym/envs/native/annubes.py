from typing import Any

import numpy as np

import neurogym as ngym
from neurogym import TrialEnv


class AnnubesEnv(TrialEnv):
    """General class for the Annubes type of tasks.

    The probabilities for different sensory modalities are specified using a dictionary used in the task.
    Note that the probabilities for the different modalities are interpreted as being relative to each other.
    For instance, `{"v": 0.25, "a": 0.75}` means that `a` would occur three times as often as `v`.

    Furthermore, note that the probability of catch trials is given separately (cf. `catch_prob` below).
    For instance, if the catch probability is `0.5`, the stimulus probabilities will be
    used only in the remaining (`1 - catch_prob`) of the trials.

    TODO: Move everything below to __init__.

    Args:
        session: Configuration of the trials that can appear during a session.
            Defaults to {"v": 0.5, "a": 0.5}.
        stim_intensities: A dictionary of stimulus types mapped to possible intensity values of each stimulus
            that is actually presented to the agent with non-zero probability. Defaults to None.
        stim_time: Duration of each stimulus in ms.
            Defaults to 1000.
        catch_prob: Probability of catch trials in the session. Must be between 0 and 1 (inclusive).
            Defaults to 0.5.
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

    Raises:
        ValueError: Raised if not all stimulus types have assigned intensities.
    """

    def __init__(
        self,
        session: dict[str | tuple[str, ...], float] | None = None,
        stim_intensities: dict[str, list[float]] | None = None,
        stim_time: int = 1000,
        catch_prob: float = 0.5,
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
        # Ensure that the session probabilities are sane.
        # A session is a specification for presenting
        # different sensory modalities with certain probabilities.
        # ==================================================
        (session, stim_types) = self._prepare_session(session)

        # Ensure that we have the elementary stimulus types in the session,
        # even if they haven't been specified explicitly.
        session.update({st: 0.0 for st in stim_types if st not in session})

        # Ensure that the stimulus intensities are specified
        # for each stimulus type.
        # Here, we use the set of elementary stimulus types
        # because they might not be specified explicitly
        # in the session, for instance if we have session = {('a', 'v'): 1}
        # ==================================================
        if stim_intensities is None:
            stim_intensities = {k: [0.8, 0.9, 1.0] for k in stim_types}

        if set(stim_intensities) != set(stim_types):
            msg = "Please ensure that all stimulus types have corresponding intensities."
            raise ValueError(msg)

        # Checks on the output behaviour and intensities
        # ==================================================
        if output_behavior is None:
            output_behavior = [0, 1]

        # Ensure that the max_sequential dictionary is
        # populated with values that make sense.
        # ==================================================
        max_sequential = self._prepare_max_sequential(stim_types, catch_prob, session, max_sequential)

        # Initialisation
        # ==================================================
        super().__init__(dt=dt)
        self.session = session
        self.stim_types = stim_types
        self.stim_intensities = stim_intensities
        self.stim_time = stim_time
        self.catch_prob = catch_prob
        self.max_sequential = max_sequential
        self.sequential_count = dict.fromkeys(max_sequential, 0)
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
            **{trial: i for i, trial in enumerate(sorted(stim_types), 2)},
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

        # Compile a dictionary of available stimulus options.
        options: dict[str | tuple[str, ...] | None, float] = {}
        for _stim_types, prob in self.session.items():
            # Ensure that the key is a tuple so that we can iterate over it.
            key = (_stim_types,) if not isinstance(_stim_types, tuple) else _stim_types

            if prob > 0.0 and all(
                (
                    self.max_sequential[stim_type] is None
                    or self.sequential_count[stim_type] < self.max_sequential[stim_type]  # type: ignore[operator]
                )
                for stim_type in key
            ):
                options[_stim_types] = (1 - self.catch_prob) * prob

        # If a catch trial is allowed, add the probability here
        if self.catch_prob > 0.0 and (
            self.max_sequential[None] is None or self.sequential_count[None] < self.max_sequential[None]  # type: ignore[operator]
        ):
            options[None] = self.catch_prob

        # Select a stimulus type from the available options.
        stim_types, catch = self._pick_stim_types(options)

        if catch:
            # Set all the GT values to 0.
            self.set_groundtruth(0, period="fixation")
            self.set_groundtruth(0, period="stimulus")
            self.set_groundtruth(0, period="iti")

            # Reset the sequential count for all modalities
            # since a catch trial breaks any sequences thereof.
            for stim_type in self.sequential_count:
                if stim_type is not None:
                    self.sequential_count[stim_type] = 0
            self.sequential_count[None] += 1

            stim_types = []
            stim_values: dict[str, float] = {}

        else:
            # Update the sequential count for the modalities that were selected.
            for stim_type in self.sequential_count:
                if stim_type in stim_types:
                    self.sequential_count[stim_type] += 1
                else:
                    self.sequential_count[stim_type] = 0

            # Now we choose the values for the modalities
            stim_values = {k: self._rng.choice(self.stim_intensities[k], 1, False)[0] for k in stim_types}

            # Set the GT
            for mod in self.stim_types:
                if mod in stim_types:
                    self.add_ob(stim_values[mod], "stimulus", where=mod)
                    # Add noise to the
                    self.add_randn(0, self.noise_factor, "stimulus", where=mod)
                # REVIEW: Should this be outside the `for` loop?
                self.set_groundtruth(0, period="fixation")
                self.set_groundtruth(1, period="stimulus")
                self.set_groundtruth(0, period="iti")

        # Build the trial
        self.trial = {
            "catch": catch,
            "stim_types": stim_types,
            "stim_values": stim_values,
            "sequential_count": self.sequential_count,
        }

        return self.trial

    def _prepare_max_sequential(
        self,
        stim_types: tuple,
        catch_prob: float,
        session: dict,
        max_sequential: dict | int | None,
        atol: float | None = None,
        rtol: float | None = None,
    ) -> dict[str | None, int | None]:
        """Check the satisfiability of the the max_sequential condition.

        Args:
            stim_types: A list of elementary stimulus types identified from the session.
            catch_prob: The catch probability.
            session: A session.
            max_sequential: An optional dictionary or integer specifying the
                number of consecutive stimulus presentations.
            atol: Absolute tolerance.
            rtol: Relative tolerance.

        Raises:
            ValueError: Raised if the max. sequential condition is not satisfiable.

        Returns:
            A dictionary specifying the number of consecutive stimulus presentations.
        """
        if atol is None:
            atol = float(10 * np.finfo(float).eps)
        if rtol is None:
            rtol = float(10 * np.finfo(float).eps)

        default_max_sequential = max_sequential if isinstance(max_sequential, int) else None
        if not isinstance(max_sequential, dict):
            max_sequential = {}

        # Now populate max_sequential, including for catch trials ('None' key)
        for _stim_type in stim_types:
            max_sequential.setdefault(_stim_type, default_max_sequential)
        max_sequential.setdefault(None, default_max_sequential)

        # Check if the max_sequential conditions can be satisfied at all.
        # This could happen if the catch probability is 0 and one or more
        # modalities are supposed to appear 100% of the time but
        # at the same time max_sequential imposes a limit on how many times
        # that stimulus can be presented.
        if np.isclose(catch_prob, 0.0, atol=atol, rtol=rtol):
            stim_probs = dict.fromkeys(stim_types, 0.0)
            for _stim_types, prob in session.items():
                if not isinstance(_stim_types, tuple):
                    _stim_types = (_stim_types,)
                for _stim_type in _stim_types:
                    stim_probs[_stim_type] += prob

            if any(
                max_sequential[k] is not None and np.isclose(v, 1.0, atol=atol, rtol=rtol)
                for k, v in stim_probs.items()
            ):
                msg = "Invalid settings: max_sequential imposes a limit \
                    on a stimulus that should appear in every trial."
                raise ValueError(msg)

        return max_sequential

    def _prepare_session(
        self,
        session: dict[str | tuple[str, ...], float] | None,
    ) -> tuple[dict[str | tuple[str, ...], float], tuple[str, ...]]:
        """Create a properly formatted session with probabilities summing up to 1.

        Args:
            session: The requested session combinations with their probabilities.

        Raises:
            ValueError: Raised if probabilities are not between 0 and 1, inclusive.
            ValueError: Raised if all probabilities specified in the session are 0.

        Returns:
            A dictionary containing the modalities (and any combinations thereof)
            with their respective probabilities.
        """

        # Helper functions
        # ==================================================
        def _tuple_or_string(item: str | tuple[str, ...]) -> str | tuple[str, ...]:
            if item is None or isinstance(item, str):
                # Not a container
                return item
            if len(item) == 1:
                # A container with one element.
                return item[0]
            # A container with more than one element.
            # Unpack nested tuples and combine into a single tuple without repetitions.
            combined = set()
            for elem in item:
                combined.update(set(elem if isinstance(elem, tuple) else (elem,)))
            return tuple(sorted(combined))

        # / Helper functions
        # ==================================================

        if session is None or not isinstance(session, dict) or len(session) == 0:
            session = {"a": 0.5, "v": 0.5}

        if any(not (0.0 <= s <= 1.0) for s in session.values()):
            msg = "Please ensure that all probabilities are between 0 and 1, inclusive."
            raise ValueError(msg)

        probability_sum = sum(session.values())

        # Ensure that at least one probability is nonzero.
        if probability_sum < np.finfo(float).eps:
            msg = "Please ensure that at least one modality has a non-zero probability."
            raise ValueError(msg)

        # Extract the elementary modalities.
        stim_types = _tuple_or_string(tuple(session.keys()))  # type: ignore[arg-type]

        # Normalise the probabilities and standardise the keys
        # (sort and ensure that there are no repetitions).
        # ==================================================
        _session = {_tuple_or_string(k): v / probability_sum for k, v in session.items()}

        return _session, stim_types  # type: ignore[return-value]

    def _pick_stim_types(
        self,
        options: dict,
    ) -> tuple[list[str], bool]:
        """Pick sitmulus types and the corrsponding intensities.

        Args:
            options: The session options to choose from.

        Returns:
            A list of stimulus types.
        """
        # Pick one of all mutually exclusive options.
        # ==================================================
        # Convert the dictionary into a list.
        # We are going to pick a random element from the array
        # rather than a random key because NumPy's random choice
        # function struggles when the elements are tuples.
        total = sum(options.values())
        kv_arr = [(k, v / total) for k, v in options.items()]

        # Select an index at random
        idx = self._rng.choice(
            np.arange(len(kv_arr)),
            1,
            False,
            p=[item[1] for item in kv_arr],
        )[0]

        # Now pick the key from the array
        stim_types = kv_arr[idx][0]

        if stim_types is None:
            # Catch trial
            return [], True
        if isinstance(stim_types, tuple):
            # Ensure that we return a list of stimulus types.
            return sorted(stim_types), False
        return [stim_types], False

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
