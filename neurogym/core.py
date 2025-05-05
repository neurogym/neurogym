import contextlib
import warnings
from typing import Any, NoReturn

import gymnasium as gym
import numpy as np

from neurogym.utils.ngym_random import trunc_exp

METADATA_DEF_KEYS = ["description", "paper_name", "paper_link", "timing", "tags"]

OBNOW = "ob_unknown_yet"  # TODO: temporary hack to create constant placeholder


def _clean_string(string):
    return " ".join(string.replace("\n", "").split())


def env_string(env, short=False):
    if short:
        return f"<{type(env).__name__}>"

    string = ""
    metadata = env.metadata
    docstring = env.__doc__
    string += f"### {type(env).__name__}\n"
    paper_name = metadata.get("paper_name", None) or "Missing paper name"
    paper_name = _clean_string(paper_name)
    paper_link = metadata.get("paper_link", None)
    string += f"Doc: {docstring}\n"
    string += "Reference paper \n"
    if paper_link is None:
        string += f"{paper_name}\n"
        string += "Missing paper link\n"
    else:
        string += f"[{paper_name}]({paper_link})\n"
    # TODO: Add timing info back? # commented out code for this TODO removed in PR #16

    if env.rewards is not None:  # if env.rewards is an array, if env.rewards will throw an error
        string += "\nReward structure \n"
        try:  # if the reward structure is a dictionary
            for key, val in env.rewards.items():
                string += f"{key} : {val}\n"
        except AttributeError:  # otherwise just add the reward structure to the string?
            string += str(env.rewards)

    # add extra info
    other_info = list(set(metadata.keys()) - set(METADATA_DEF_KEYS))
    if len(other_info) > 0:
        string += "\nOther parameters: \n"
        for key in other_info:
            string += f"{key} : {_clean_string(str(metadata[key]))}\n"
    # tags
    if "tags" in metadata:
        tags = metadata["tags"]
        string += "\nTags: "
        for tag in tags:
            string += f"{tag}, "
        string = string[:-2] + ".\n"

    return string


class BaseEnv(gym.Env):
    """The base Neurogym class to include dt."""

    def __init__(self, dt=100) -> None:
        super().__init__()
        self.dt = dt
        self.t = self.t_ind = 0
        self.tmax = 10000  # maximum time steps
        self.performance = 0
        self.rewards: dict = {}
        self.rng = np.random.RandomState()

    def seed(self, seed=None):
        """Set random seed."""
        self.rng = np.random.RandomState(seed)
        if self.action_space is not None:
            self.action_space.seed(seed)
        return [seed]


class TrialEnv(BaseEnv):
    """The main Neurogym class for trial-based envs."""

    def __init__(self, dt=100, num_trials_before_reset=10000000, r_tmax=0) -> None:
        super().__init__(dt=dt)
        self.r_tmax = r_tmax
        self.num_tr = 0
        self.num_tr_exp = num_trials_before_reset
        self.trial: dict | None = None
        self._ob_built = False
        self._gt_built = False
        self._has_gt = False  # check if the task ever defined gt

        self._default_ob_value = None  # default to 0

        # For optional periods
        self.timing: dict = {}
        self.start_t: dict = {}
        self.end_t: dict = {}
        self.start_ind: dict = {}
        self.end_ind: dict = {}
        self._tmax = 0  # Length of each trial

        self._top = self
        self._duration: dict = {}

    def __str__(self) -> Any:
        """Information about task."""
        return env_string(self, short=True)

    def _new_trial(self, **kwargs) -> NoReturn:
        """Private interface for starting a new trial.

        Returns:
            trial: dict of trial information. Available to step function as
                self.trial
        """
        msg = "_new_trial is not defined by user."
        raise NotImplementedError(msg)

    def _step(self, _action) -> NoReturn:
        """Private interface for the environment.

        Receives an action and returns a new state, a reward, a flag variable
        indicating whether the experiment has ended and a dictionary with
        useful information.
        """
        msg = "_step is not defined by user."
        raise NotImplementedError(msg)

    def seed(self, seed=None):
        """Set random seed."""
        self.rng = np.random.RandomState(seed)
        if hasattr(self, "action_space") and self.action_space is not None:
            self.action_space.seed(seed)
        for val in self.timing.values():
            with contextlib.suppress(AttributeError):
                val.seed(seed)
        return [seed]

    @staticmethod
    def post_step(ob, reward, terminated, truncated, info):
        """Optional task-specific wrapper applied at the end of step.

        It allows to modify ob online (e.g. provide a specific observation for different actions made by the agent)
        """
        return ob, reward, terminated, truncated, info

    def new_trial(self, **kwargs):
        """Public interface for starting a new trial.

        Returns:
            trial: dict of trial information. Available to step function as
                self.trial
        """
        # Reset for next trial
        self._tmax = 0  # reset, self.tmax not reset so it can be used in step
        self._ob_built = False
        self._gt_built = False
        trial = self._new_trial(**kwargs)
        self.trial = trial
        self.num_tr += 1  # Increment trial count
        self._has_gt = self._gt_built
        return trial

    def step(self, action):
        """Public interface for the environment."""
        ob, reward, terminated, truncated, info = self._step(action)

        if "new_trial" not in info:
            info["new_trial"] = False

        if self._has_gt and "gt" not in info:
            # If gt is built, default gt to gt_now
            # must run before incrementing t
            info["gt"] = self.gt_now

        self.t += self.dt  # increment within trial time count
        self.t_ind += 1

        if self.t + self.dt > self.tmax and not info["new_trial"]:
            info["new_trial"] = True
            reward += self.r_tmax

        # TODO: new_trial happens after step, so trial indx precedes obs change
        if info["new_trial"]:
            info["performance"] = self.performance
            self.t = self.t_ind = 0  # Reset within trial time count
            trial = self._top.new_trial()
            self.performance = 0
            info["trial"] = trial
        if ob is OBNOW:
            ob = self.ob[self.t_ind]
        return self.post_step(ob, reward, terminated, truncated, info)

    def reset(self, seed=None, options=None):
        """Reset the environment.

        Args:
            seed: random seed, overwrites self.seed if not None
            options: additional options used to reset the env.
                Can include 'step_fn' and 'no_step'.
                `step_fn` can be a function or None. If function, overwrite original
                `self.step` method.
                `no_step` is a bool. If True, no step is taken and observation randomly
                sampled. It defaults to False.
        """
        super().reset(seed=seed)

        self.num_tr = 0
        self.t = self.t_ind = 0

        step_fn = options.get("step_fn") if options else None
        no_step = options.get("no_step", False) if options else False

        self._top.new_trial()

        # have to also call step() to get the initial ob since some wrappers modify step() but not new_trial()
        self.action_space.seed(0)
        if no_step:
            return self.observation_space.sample(), {}
        if step_fn is None:
            ob, _, _, _, _ = self._top.step(self.action_space.sample())
        else:
            ob, _, _, _, _ = step_fn(self.action_space.sample())
        return ob, {}

    def render(self, mode="human") -> None:
        """Plots relevant variables/parameters."""

    def set_top(self, wrapper) -> None:
        """Set top to be wrapper."""
        self._top = wrapper

    def sample_time(self, period):
        timing = self.timing[period]
        if isinstance(timing, int | float):
            t = timing
        elif callable(timing):
            t = timing()
        elif isinstance(timing[0], int | float):
            # Expect list of int/float, and use random choice
            t = self.rng.choice(timing)
        else:
            dist, args = timing
            if dist == "uniform":
                t = self.rng.uniform(*args)
            elif dist == "choice":
                t = self.rng.choice(args)
            elif dist == "truncated_exponential":
                t = trunc_exp(self.rng, *args)
            elif dist == "constant":
                t = args
            elif dist == "until":
                # set period duration such that self.t_end[period] = args
                t = args - self.tmax
                if t < 0:
                    msg = (
                        f"Invalid 'until' time for period {period}. Current max time: {self.tmax},",
                        f"Requested end time: {args}",
                    )
                    raise ValueError(msg)
            else:
                msg = f"Distribution {dist} not found."
                raise ValueError(msg)
        return (t // self.dt) * self.dt

    def add_period(
        self,
        period,
        duration=None,
        before=None,
        after=None,
        last_period=False,
    ) -> None:
        """Add an period.

        Args:
            period: string or list of strings, name of the period
            duration: float or None, duration of the period
                if None, inferred from timing_fn
            before: (optional) str, name of period that this period is before
            after: (optional) str, name of period that this period is after
                or float, time of period start
            last_period: bool, default False. If True, then this is last period
                will generate self.tmax, self.tind, and self.ob
        """
        if self._ob_built:
            msg = "Cannot add period after ob is built, i.e. after running add_ob."
            raise InvalidOperationError(msg)
        if isinstance(period, str):
            pass
        else:
            if duration is None:
                duration = [None] * len(period)
            elif len(duration) != len(period):
                msg = f"{len(duration)=} and {len(period)=} must be the same."
                raise ValueError(msg)

            # Recursively calling itself
            self.add_period(period[0], duration=duration[0], after=after)
            for i in range(1, len(period)):
                is_last = (i == len(period) - 1) and last_period
                self.add_period(
                    period[i],
                    duration=duration[i],
                    after=period[i - 1],
                    last_period=is_last,
                )
            return

        if duration is None:
            duration = self.sample_time(period)
        self._duration[period] = duration

        if after is not None:
            start = self.end_t[after] if isinstance(after, str) else after
        elif before is not None:
            start = self.start_t[before] - duration
        else:
            start = 0  # default start with 0

        self.start_t[period] = start
        self.end_t[period] = start + duration
        self.start_ind[period] = int(start / self.dt)
        self.end_ind[period] = int((start + duration) / self.dt)

        self._tmax = max(self._tmax, start + duration)
        self.tmax = int(self._tmax / self.dt) * self.dt

    def _init_ob(self) -> None:
        """Initialize trial info with tmax, tind, ob."""
        tmax_ind = int(self._tmax / self.dt)
        if self.observation_space.shape is None:
            msg = "observation_space.shape cannot be None"
            raise ValueError(msg)
        ob_shape = [tmax_ind, *list(self.observation_space.shape)]
        if self._default_ob_value is None:
            self.ob = np.zeros(ob_shape, dtype=self.observation_space.dtype)
        else:
            self.ob = np.full(
                ob_shape,
                self._default_ob_value,
                dtype=self.observation_space.dtype,
            )
        self._ob_built = True

    def _init_gt(self) -> None:
        """Initialize trial with ground_truth."""
        tmax_ind = int(self._tmax / self.dt)
        if self.action_space.shape is None:
            msg = "action_space.shape cannot be None"
            raise ValueError(msg)
        self.gt = np.zeros(
            [tmax_ind, *list(self.action_space.shape)],
            dtype=self.action_space.dtype,
        )
        self._gt_built = True

    def view_ob(self, period=None):
        """View observation of an period."""
        if not self._ob_built:
            self._init_ob()

        if period is None:
            return self.ob
        return self.ob[self.start_ind[period] : self.end_ind[period]]

    def _add_ob(self, value, period=None, where=None, reset=False) -> None:
        """Set observation in period to value.

        Args:
            value: number, numpy array, or callable
                If number: broadcast to full observation shape
                If array: must match observation space dimensions (observation_space.shape[0], ...)
                If callable: takes observation array as input and returns array of the same shape
            period: string, must be the name of an added period
            where: string or np array, location of stimulus to be added
            reset: bool, whether to zero-out values before adding
        """
        if isinstance(period, str) or period is None:
            period = [period]
        if isinstance(where, str):
            where = self.observation_space.name[where]  # type: ignore[attr-defined]

        for p in period:
            ob = self.view_ob(period=p)

            if where is None:
                if reset:
                    ob *= 0
                ob += value(ob) if callable(value) else value
            else:
                if reset:
                    ob[..., where] *= 0
                # TODO: This only works if the slicing is one one-dimension
                ob[..., where] += value(ob[..., where]) if callable(value) else value

    def add_ob(self, value, period=None, where=None) -> None:
        """Add value to observation.

        Args:
            value: array-like (ob_space.shape, ...)
            period: string, must be name of an added period
            where: string or np array, location of stimulus to be added
        """
        self._add_ob(value, period, where, reset=False)

    def add_randn(self, mu=0, sigma=1, period=None, where=None) -> None:
        if isinstance(period, str) or period is None:
            pass
        else:
            for p in period:
                self.add_randn(mu, sigma, p, where)
            return

        ob = self.view_ob(period=period)
        if where is None:
            ob += mu + self.rng.randn(*ob.shape) * sigma
        else:
            if isinstance(where, str):
                where = self.observation_space.name[where]  # type: ignore[attr-defined]
            # TODO: This only works if the slicing is one one-dimension
            ob[..., where] += mu + self.rng.randn(*ob[..., where].shape) * sigma

    def set_ob(self, value, period=None, where=None) -> None:
        self._add_ob(value, period, where, reset=True)

    def set_groundtruth(self, value, period=None, where=None) -> None:
        """Set groundtruth value."""
        if not self._gt_built:
            self._init_gt()

        if where is not None:
            # TODO: Only works for Discrete action_space, make it work for Box
            value = self.action_space.name[where][value]  # type: ignore[attr-defined]
        if isinstance(period, str):
            self.gt[self.start_ind[period] : self.end_ind[period]] = value
        elif period is None:
            self.gt[:] = value
        else:
            for p in period:
                self.set_groundtruth(value, p)

    def view_groundtruth(self, period):
        """View observation of an period."""
        if not self._gt_built:
            self._init_gt()
        return self.gt[self.start_ind[period] : self.end_ind[period]]

    def in_period(self, period, t=None):
        """Check if current time or time t is in period."""
        if t is None:
            t = self.t  # Default
        return self.start_t[period] <= t < self.end_t[period]

    @property
    def ob_now(self):
        return OBNOW

    @property
    def gt_now(self):
        return self.gt[self.t_ind]

    def trial_length_stats(self, num_trials: int = 10000) -> dict:
        """Calculate statistics about trial lengths.

        Args:
            num_trials (int, optional): Number of trials to sample. Defaults to 10000.

        Returns:
            dict: Contains 'mean', 'std', 'percentile_95', and 'max' as floats.
        """
        # For environments with very simple timing (all fixed durations),
        # we can calculate exactly without sampling
        if self.timing and all(isinstance(timing, int | float) for timing in self.timing.values()):
            fixed_length = int(sum(self.timing.values()) / self.dt)
            return {
                "mean": fixed_length,
                "std": 0,
                "percentile_95": fixed_length,
                "max": fixed_length,
            }

        # For more complex environments, we sample trials
        trial_lengths_list = []

        # Store current RNG state to restore later
        rng_state = self.rng.get_state()

        # Sample trials
        for _ in range(num_trials):
            self.new_trial()
            if hasattr(self, "ob") and self.ob is not None:
                trial_lengths_list.append(self.ob.shape[0])

        # Restore RNG state
        self.rng.set_state(rng_state)

        # Calculate statistics from sampled trials
        if len(trial_lengths_list) == 0:
            warnings.warn("No trials were sampled. Returning default values.", stacklevel=2)
            return {
                "mean": 0,
                "std": 0,
                "percentile_95": 0,
                "max": 0,
            }
        trial_lengths = np.array(trial_lengths_list)
        return {
            "mean": round(np.mean(trial_lengths), 3),
            "std": round(np.std(trial_lengths), 3),
            "percentile_95": round(np.percentile(trial_lengths, 95), 3),
            "max": round(np.max(trial_lengths), 3),
        }


class TrialWrapper(gym.Wrapper):
    """Base class for wrapping TrialEnv."""

    def __init__(self, env) -> None:
        super().__init__(env)
        self.env = env
        if not isinstance(self.unwrapped, TrialEnv):
            msg = f"Trial wrapper must be used on TrialEnv, not {self.unwrapped}."
            raise TypeError(msg)
        self.unwrapped.set_top(self)

    @property
    def task(self):
        """Alias."""
        return self.unwrapped

    def new_trial(self, **kwargs) -> NoReturn:
        raise NotImplementedError


class InvalidOperationError(Exception):
    """Raised when an operation is not allowed."""
