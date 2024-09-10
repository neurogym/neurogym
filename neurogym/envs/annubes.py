import numpy as np

import neurogym as ngym
from neurogym import TrialEnv


class AnnubesEnv(TrialEnv):
    def __init__(
        self,
        session=None,
        stim_intensities=None,
        stim_time=1000,
        catch_prob=0.5,
        fix_intensity=0,
        fix_time=500,
        dt=100,
        tau=100,
        n_outputs=2,
        output_behavior=None,
        noise_std=0.01,
        reward_dict=None,
        random_seed=None,
    ):
        if output_behavior is None:
            output_behavior = [0, 1]
        if stim_intensities is None:
            stim_intensities = [0.8, 0.9, 1.0]
        if session is None:
            session = {"v": 0.5, "a": 0.5}
        super().__init__(dt=dt)

        self.session = session
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
            random_seed = rng.integers(2**32)
        self._rng = np.random.default_rng(random_seed)
        self._random_seed = random_seed
        # Rewards
        if reward_dict is None:
            self.rewards = {"abort": -0.1, "correct": +1.0, "fail": 0.0}
        else:
            self.rewards = reward_dict
        self.timing = {"fixation": self.fix_time, "stimulus": self.stim_time}
        # Set the name of each input dimension
        obs_space_name = {"fixation": 0, "start": 1, "v": 2, "a": 3}
        self.observation_space = ngym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(obs_space_name),),
            dtype=np.float32,
            name=obs_space_name,
        )
        # Set the name of each action value
        self.action_space = ngym.spaces.Discrete(
            self.n_outputs,
            name={"fixation": self.output_behavior[0], "choice": self.output_behavior},
        )

    def _new_trial(self):
        # Reset trial-related variables
        self.trial = {}

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

        # Trial information
        self.trial = {
            "catch": catch,
            "stim_type": stim_type,
            "stim_value": stim_value,
        }

        return self.trial

    def _step(self, action):
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
