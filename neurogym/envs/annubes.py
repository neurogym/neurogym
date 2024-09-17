import numpy as np

import neurogym as ngym
from neurogym import TrialEnv


class AnnubesEnv(TrialEnv):
    """General class for the Annubes type of tasks.

    Args:
        session: Configuration of the trials that can appear during a session.
            It is given by a dictionary representing the ratio (values) of the different trials (keys) within the task.
            Trials with a single modality (e.g., a visual trial) must be represented by single characters, while trials
            with multiple modalities (e.g., an audiovisual trial) are represented by the character combination of those
            trials. Note that values are read relative to each other, such that e.g. `{"v": 0.25, "a": 0.75}` is
            equivalent to `{"v": 1, "a": 3}`.
            Defaults to {"v": 0.5, "a": 0.5}.
        stim_intensities: List of possible intensity values of each stimulus, when the stimulus is present. Note that
            when the stimulus is not present, the intensity is set to 0.
            Defaults to [0.8, 0.9, 1].
        stim_time: Duration of each stimulus in ms.
            Defaults to 1000.
        catch_prob: Probability of catch trials in the session. Must be between 0 and 1 (inclusive).
            Defaults to 0.5.
        fix_intensity: Intensity of input signal during fixation.
            Defaults to 0.
        fix_time: Fixation time in ms. Note that the duration of each input and output signal is increased by this time.
            Defaults to 500.
        dt: Time step in ms.
            Defaults to 100.
        tau: Time constant in ms.
            Defaults to 100.
        n_outputs: Number of possible outputs, signaling different behavioral choices.
            Defaults to 2.
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
        session: dict[str, float] | None = None,
        stim_intensities: list[float] | None = None,
        stim_time: int = 1000,
        catch_prob: float = 0.5,
        fix_intensity: float = 0,
        fix_time: int = 500,
        dt: int = 100,
        tau: int = 100,
        n_outputs: int = 2,
        output_behavior: list[float] | None = None,
        noise_std: float = 0.01,
        rewards: dict[str, float] | None = None,
        random_seed: int | None = None,
    ):
        if session is None:
            session = {"v": 0.5, "a": 0.5}
        if output_behavior is None:
            output_behavior = [0, 1]
        if stim_intensities is None:
            stim_intensities = [0.8, 0.9, 1.0]
        if session is None:
            session = {"v": 0.5, "a": 0.5}
        super().__init__(dt=dt)
        self.session = {i: session[i] / sum(session.values()) for i in session}
        self.stim_intensities = stim_intensities
        self.stim_time = stim_time
        self.catch_prob = catch_prob
        self.fix_intensity = fix_intensity
        self.fix_time = fix_time
        self.dt = dt
        self.tau = tau
        self.n_outputs = n_outputs
        self.output_behavior = output_behavior
        self.noise_std = noise_std
        self.random_seed = random_seed
        alpha = dt / self.tau
        self.noise_factor = self.noise_std * np.sqrt(2 * alpha) / alpha
        # Set random state
        if random_seed is None:
            rng = np.random.default_rng(random_seed)
            self._random_seed = rng.integers(2**32)
        self._rng = np.random.default_rng(self._random_seed)
        # Rewards
        if rewards is None:
            self.rewards = {"abort": -0.1, "correct": +1.0, "fail": 0.0}
        else:
            self.rewards = rewards
        self.timing = {"fixation": self.fix_time, "stimulus": self.stim_time}
        # Set the name of each input dimension
        obs_space_name = {"fixation": 0, "start": 1, **{trial: i for i, trial in enumerate(session, 2)}}
        self.observation_space = ngym.spaces.Box(low=0.0, high=1.0, shape=(len(obs_space_name),), name=obs_space_name)
        # Set the name of each action value
        self.action_space = ngym.spaces.Discrete(
            self.n_outputs,
            name={"fixation": self.output_behavior[0], "choice": self.output_behavior},
        )

    def _new_trial(self) -> dict:
        """Internal method to generate a new trial.

        Returns:
            A dictionary containing the information of the new trial.
        """
        # Setting time periods and their order for this trial
        self.add_period(["fixation", "stimulus"])

        # Adding fixation and start signal values
        self.add_ob(self.fix_intensity, "fixation", where="fixation")
        self.add_ob(1, "stimulus", where="start")

        # Catch trial decision
        catch = self._rng.choice([0, 1], p=[self.catch_prob, 1 - self.catch_prob])
        stim_type = None
        stim_value = None
        if not catch:
            stim_type = self._rng.choice(list(self.session.keys()), p=list(self.session.values()))
            stim_value = self._rng.choice(self.stim_intensities, 1)
            for mod in self.session:
                if stim_type == mod:
                    self.add_ob(stim_value, "stimulus", where=mod)
                    self.add_randn(0, self.noise_factor, "stimulus", where=mod)
                self.set_groundtruth(0, period="fixation")
                self.set_groundtruth(1, period="stimulus")
        else:
            self.set_groundtruth(0, period="fixation")
            self.set_groundtruth(0, period="stimulus")

        return {"catch": catch, "stim_type": stim_type, "stim_value": stim_value}

    def _step(self, action: int) -> tuple:
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

        if self.in_period("fixation"):
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
            # esch step is self.dt ms
            # self.tmax is the maximum number of time steps within a trial
            # see self.add_period in TrialEnv for more details
            if self.t >= self.tmax - self.dt:
                new_trial = True

        info = {"new_trial": new_trial, "gt": gt}
        if new_trial:
            info["trial"] = self.trial
            self.trial = {}

        return self.ob_now, reward, terminated, truncated, info