import numpy as np

import neurogym as ngym
from neurogym.utils.scheduler import RandomSchedule
from neurogym.wrappers import ScheduleEnvs

disable_env_checker = False
rng = np.random.default_rng()


def make_env(name, **kwargs):
    if disable_env_checker:
        return ngym.make(name, disable_env_checker=True, **kwargs)
    # cannot add the arg disable_env_checker to gym.make in versions lower than 0.24
    # FIXME: given that we are using gymnasium, is this still relevant?
    return ngym.make(name, **kwargs)


class CstObTrialWrapper(ngym.TrialWrapper):
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
    env = make_env(ngym.all_envs()[0])
    return CstObTrialWrapper(env, cst_ob)


def test_wrapper_new_trial():
    """Test that the ob returned by new_trial takes the wrapper correctly into account."""
    cst_ob = rng.random(10)
    env = _setup_env(cst_ob)
    env.new_trial()
    ob = env.ob[0]
    assert ob.shape == cst_ob.shape, f"Got shape {ob.shape} but expected shape {cst_ob.shape}"
    assert np.all(ob == cst_ob)


def test_wrapper_reset():
    """Test that the ob returned by reset takes the wrapper correctly into account."""
    cst_ob = rng.random(10)
    env = _setup_env(cst_ob)
    ob, _ = env.reset()

    assert ob.shape == cst_ob.shape, f"Got shape {ob.shape} but expected shape {cst_ob.shape}"
    assert np.all(ob == cst_ob)


def test_wrapper_step():
    """Test that the ob returned by step takes the wrapper correctly into account."""
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
    tasks = ngym.get_collection("yang19")
    envs = [make_env(task) for task in tasks]
    schedule = RandomSchedule(len(envs))
    env = ScheduleEnvs(envs, schedule=schedule, env_input=True)

    env.reset()
    env.step(env.action_space.sample())


def test_schedule_envs():
    tasks = ngym.get_collection("yang19")
    envs = [make_env(task) for task in tasks]
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
