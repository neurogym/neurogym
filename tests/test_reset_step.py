import warnings

import numpy as np

from neurogym.core import TrialWrapper
from neurogym.envs.collections import get_collection
from neurogym.envs.registration import all_envs, make
from neurogym.utils.logging import logger
from neurogym.utils.scheduler import RandomSchedule
from neurogym.wrappers.block import ScheduleEnvs

disable_env_checker = False
rng = np.random.default_rng()


class CstObTrialWrapper(TrialWrapper):
    def __init__(self, env, cst_ob) -> None:
        super().__init__(env)
        self.cst_ob = cst_ob

    def new_trial(self, **kwargs):
        trial = self.env.new_trial(**kwargs)
        self.ob = np.repeat(self.cst_ob[None, :], self.ob.shape[0], axis=0)
        return trial

    # modifying new_trial is not enough to modify the ob returned by step()
    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        new_ob = self.ob[self.t_ind]
        return new_ob, reward, terminated, truncated, info


def _setup_env(cst_ob):
    env_name = all_envs()[1]  # Just an example, replace with a specific env if needed
    logger.info(f"Example env used: {env_name}")
    env = make(env_name)
    return CstObTrialWrapper(env, cst_ob)


def test_wrapper_new_trial():
    """Test that the ob returned by new_trial takes the wrapper correctly into account."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        cst_ob = rng.random(10)
        env = _setup_env(cst_ob)
        env.new_trial()
        ob = env.ob[0]
        assert ob.shape == cst_ob.shape, f"Got shape {ob.shape} but expected shape {cst_ob.shape}"
        assert np.all(ob == cst_ob)


def test_wrapper_reset():
    """Test that the ob returned by reset takes the wrapper correctly into account."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        cst_ob = rng.random(10)
        env = _setup_env(cst_ob)
        ob, _ = env.reset()

        assert ob.shape == cst_ob.shape, f"Got shape {ob.shape} but expected shape {cst_ob.shape}"
        assert np.all(ob == cst_ob)


def test_wrapper_step():
    """Test that the ob returned by step takes the wrapper correctly into account."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        cst_ob = rng.random(10)
        env = _setup_env(cst_ob)
        env.reset()
        ob, _, _, _, _ = env.step(env.action_space.sample())
        assert ob.shape == cst_ob.shape, f"Got shape {ob.shape} but expected shape {cst_ob.shape}"
        assert np.all(ob == cst_ob)


def test_reset_with_scheduler():
    """Test that ScheduleEnvs.reset() resets all the environments in its list envs.

    This is required before being able to call step() (enforced by the gymnasium wrapper OrderEnforcing).
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        tasks = get_collection("yang19")
        envs = [make(task) for task in tasks]
        schedule = RandomSchedule(len(envs))
        env = ScheduleEnvs(envs, schedule=schedule, env_input=True)

        env.reset()
        env.step(env.action_space.sample())


def test_schedule_envs():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        tasks = get_collection("yang19")
        envs = [make(task) for task in tasks]
        for i, env in enumerate(envs):
            envs[i] = CstObTrialWrapper(env, np.array([i]))

        schedule = RandomSchedule(len(envs))
        env = ScheduleEnvs(envs, schedule=schedule, env_input=True)
        env.reset()
        for _ in range(5):
            env.new_trial()
            assert np.all([ob == env.i_env for ob in env.ob])
            # test rule input
            assert env.i_env == np.argmax(env.unwrapped.ob[0, -len(envs) :])
