from typing import Any

import numpy as np
from gymnasium import spaces

from neurogym.core import TrialWrapper


class RandomGroundTruth(TrialWrapper):
    # TODO: A better name?

    def __init__(self, env, p=None) -> None:
        super().__init__(env)
        try:
            self.n_ch = len(self.choices)  # max num of choices
        except AttributeError as e:
            msg = "RandomGroundTruth requires task to have attribute choices."
            raise AttributeError(msg) from e
        if p is None:
            p = np.ones(self.n_ch) / self.n_ch
        self.p = p

    def new_trial(self, **kwargs):
        p = kwargs.get("p", self.p)
        ground_truth = self.rng.choice(self.env.choices, p=p)  # type: ignore[attr-defined]
        kwargs = {"ground_truth": ground_truth}
        return self.env.new_trial(**kwargs)  # type: ignore[attr-defined]


class ScheduleAttr(TrialWrapper):
    """Schedule attributes.

    Args:
        env: TrialEnv object
        schedule:
    """

    def __init__(self, env, schedule, attr_list) -> None:
        super().__init__(env)
        self.schedule = schedule
        self.attr_list = attr_list

    def seed(self, seed=None) -> None:
        self.schedule.seed(seed)
        self.env.seed(seed)  # type: ignore[attr-defined]

    def new_trial(self, **kwargs):
        i = self.schedule()
        kwargs.update(self.attr_list[i])
        return self.env.new_trial(**kwargs)  # type: ignore[attr-defined]


def _have_equal_shape(envs) -> None:
    """Check if environments have equal shape."""
    env_ob_shape = envs[0].observation_space.shape
    for env in envs:
        if env.observation_space.shape != env_ob_shape:
            msg = (
                "Env must have equal observation shape.",
                f"Instead got {env.observation_space.shape}) for {env} and {env_ob_shape} for {envs[0]}",
            )
            raise ValueError(msg)

    env_act_shape = envs[0].action_space.n
    for env in envs:
        if env.action_space.n != env_act_shape:
            msg = (
                "Env must have equal action shape.",
                f"Instead got {env.action_space.n}) for {env} and {env_act_shape} for {envs[0]}",
            )
            raise ValueError(msg)


class MultiEnvs(TrialWrapper):
    """Wrap multiple environments.

    Args:
        envs: list of env object
        env_input: bool, if True, add scalar inputs indicating current
            envinronment. default False.
    """

    def __init__(self, envs, env_input=False) -> None:
        super().__init__(envs[0])
        for env in envs:
            env.unwrapped.set_top(self)
        self.envs = envs
        self.i_env = 0

        self.env_input = env_input
        if env_input:
            env_shape = envs[0].observation_space.shape
            if len(env_shape) > 1:
                msg = f"Env must have 1-D Box shape but got {env_shape}."
                raise ValueError(msg)
            _have_equal_shape(envs)
            self.observation_space: spaces.Box = spaces.Box(
                -np.inf,
                np.inf,
                shape=(env_shape[0] + len(self.envs),),
                dtype=self.envs[0].observation_space.dtype,
            )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        reset_kwargs: dict[str, Any] = {}

        if seed is not None:
            reset_kwargs["seed"] = seed
        if options is not None:
            reset_kwargs.update(options)

        ob = super().reset(**reset_kwargs)

        return ob, {}

    def set_i(self, i) -> None:
        """Set the i-th environment."""
        self.i_env = i
        self.env = self.envs[self.i_env]

    def new_trial(self, **kwargs):
        if not self.env_input:
            return self.env.new_trial(**kwargs)

        trial = self.env.new_trial(**kwargs)
        # Expand observation
        env_ob = np.zeros(
            (self.unwrapped.ob.shape[0], len(self.envs)),  # type: ignore[attr-defined]
            dtype=self.unwrapped.ob.dtype,  # type: ignore[attr-defined]
        )
        env_ob[:, self.i_env] = 1.0
        self.unwrapped.ob = np.concatenate((self.unwrapped.ob, env_ob), axis=-1)  # type: ignore[attr-defined]
        return trial


# TODO: EnvsWrapper or MultiEnvWrapper
class ScheduleEnvs(TrialWrapper):
    """Schedule environments.

    Args:
        envs: list of env object
        schedule: utils.scheduler.BaseSchedule object
        env_input: bool, if True, add scalar inputs indicating current
            environment. default False.
    """

    def __init__(self, envs, schedule, env_input=False) -> None:
        super().__init__(envs[0])
        for env in envs:
            env.unwrapped.set_top(self)
        self.envs = envs
        self.schedule = schedule
        self.i_env = self.next_i_env = 0

        self.env_input = env_input
        if env_input:
            env_shape = envs[0].observation_space.shape
            if len(env_shape) > 1:
                msg = f"Env must have 1-D Box shape but got {env_shape}."
                raise ValueError(msg)
            _have_equal_shape(envs)
            self.observation_space: spaces.Box = spaces.Box(
                -np.inf,
                np.inf,
                shape=(env_shape[0] + len(self.envs),),
                dtype=np.float32,
            )

    def seed(self, seed=None) -> None:
        for env in self.envs:
            env.seed(seed)
        self.schedule.seed(seed)

    def reset(self, **kwargs):
        # TODO: kwargs to specify the condition for new_trial
        """Resets environments.

        Reset each environment in self.envs and use the scheduler to select the environment returning
        the initial observation. This environment is also used to set the current environment self.env.
        """
        self.schedule.reset()
        return_i_env = self.schedule()

        # first reset all the env excepted return_i_env
        for i, env in enumerate(self.envs):
            if i == return_i_env:
                continue

            # change the current env so that calling _top.new_trial() in env.reset() will generate a trial for the env
            # being currently reset (and not an env that is not yet reset)
            self.set_i(i)
            # same env used here and in the first call to new_trial()
            self.next_i_env = self.i_env

            env.reset(**kwargs)

        # then reset return_i_env and return the result
        self.set_i(return_i_env)
        self.next_i_env = self.i_env
        return self.env.reset()

    def new_trial(self, **kwargs):
        # self.env has to be changed at the beginning of new_trial, not at the end
        # but don't use schedule here since don't want to change the env between reset() and first call to new_trial()
        self.i_env = self.next_i_env
        self.env = self.envs[self.i_env]

        if not self.env_input:
            trial = self.env.new_trial(**kwargs)
        else:
            trial = self.env.new_trial(**kwargs)
            # Expand observation
            env_ob = np.zeros(
                (self.unwrapped.ob.shape[0], len(self.envs)),  # type: ignore[attr-defined]
                dtype=self.unwrapped.ob.dtype,  # type: ignore[attr-defined]
            )
            env_ob[:, self.i_env] = 1.0
            self.unwrapped.ob = np.concatenate((self.unwrapped.ob, env_ob), axis=-1)  # type: ignore[attr-defined]

        self.next_i_env = self.schedule()
        if self.env != self.envs[self.i_env]:
            msg = "want self.ob to refer to the ob of the new trial, so can't change self.env here => use next_i_env"
            raise ValueError(msg)  # FIXME: unclear error message
        return trial

    def set_i(self, i) -> None:
        """Set the current environment to the i-th environment in the list envs."""
        self.i_env = i
        self.env = self.envs[self.i_env]
        self.schedule.i = i

    def __str__(self) -> str:
        string = f"<{type(self).__name__}"
        for env in self.envs:
            for line in str(env).splitlines():
                string += f"\n\t{line}"
        string += "\n>"
        return string


class TrialHistoryV2(TrialWrapper):
    """Change ground truth probability based on previous outcome.

    Args:
        probs: matrix of probabilities of the current choice conditioned
            on the previous. Shape, num-choices x num-choices
    """

    def __init__(self, env, probs=None) -> None:
        super().__init__(env)
        try:
            self.n_ch = len(self.choices)  # max num of choices
        except AttributeError as e:
            msg = "TrialHistory requires task to have attribute choices."
            raise AttributeError(msg) from e
        if probs is None:
            probs = np.ones((self.n_ch, self.n_ch)) / self.n_ch  # uniform
        self.probs = probs
        if self.probs.shape != (self.n_ch, self.n_ch):
            msg = f"{self.probs.shape=} should be {self.n_ch, self.n_ch=}."
            raise ValueError(msg)
        self.prev_trial = self.rng.choice(self.n_ch)  # random initialization

    def new_trial(self, **kwargs):
        probs = kwargs.get("probs", self.probs)
        p = probs[self.prev_trial, :]
        # Choose ground truth and update previous trial info
        self.prev_trial = self.rng.choice(self.n_ch, p=p)
        ground_truth = self.choices[self.prev_trial]
        kwargs.update({"ground_truth": ground_truth, "probs": probs})
        return self.env.new_trial(**kwargs)  # type: ignore[attr-defined]
