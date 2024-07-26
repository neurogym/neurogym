import warnings

import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt

import neurogym as ngym
from neurogym.utils.data import Dataset

try:
    import psychopy  # noqa: F401

    _have_psychopy = True  # FIXME should psychopy be always tested, to ensure CI doesn't fail?
except ImportError:
    _have_psychopy = False

ENVS = ngym.all_envs(psychopy=_have_psychopy, contrib=True, collections=True)
# Envs without psychopy, TODO: check if contrib or collections include psychopy
ENVS_NOPSYCHOPY = ngym.all_envs(psychopy=False, contrib=True, collections=True)
SEED = 0


def _test_run(env, num_steps=100, verbose=False):
    """Test if one environment can at least be run."""
    if isinstance(env, str):
        env = ngym.make(env)
    elif not isinstance(env, gym.Env):
        msg = f"{type(env)=} must be a string or a gym.Env"
        raise TypeError(msg)

    env.reset()
    for _ in range(num_steps):
        action = env.action_space.sample()
        _state, _rew, terminated, _truncated, _info = env.step(action)
        if terminated:
            env.reset()

    tags = env.metadata.get("tags", [])
    all_tags = ngym.all_tags()
    for t in tags:
        if t not in all_tags:
            print(f"Warning: env has tag {t} not in all_tags")

    if verbose:
        print(env)

    return env


def test_run_all(verbose_success=False):
    """Test if all environments can at least be run."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        assert ngym.all_envs()[0] in ENVS
        for env_name in ENVS:
            print(env_name)
            _test_run(env_name, verbose=verbose_success)


def _test_dataset(env):
    """Main function for testing if an environment is healthy."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")

        print("Testing Environment:", env)
        kwargs = {"dt": 20}
        dataset = Dataset(env, env_kwargs=kwargs, batch_size=16, seq_len=300, cache_len=1e4)
        for _ in range(10):
            inputs, target = dataset()
            assert inputs.shape[0] == target.shape[0]


def test_dataset_all():
    """Test if all environments can at least be run."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Casting input x to numpy array.*")
        warnings.filterwarnings("ignore", message=".*is not within the observation space*")
        warnings.filterwarnings("ignore", message=".*method was expecting numpy array dtype to be*")
        warnings.filterwarnings("ignore", message=".*method was expecting a numpy array*")
        success_count = 0
        total_count = len(ngym.all_envs())
        supervised_count = len(ngym.all_envs(tag="supervised"))
        for env_name in ngym.all_envs():
            print(f"Running env: {env_name}")
            try:  # FIXME, tests are not actually performed here, as any error is caught away
                _test_dataset(env_name)
                print("Success")
                success_count += 1
            except BaseException as e:  # noqa: BLE001 # FIXME: unclear which error is expected here.
                print(f"Failure at running env: {env_name}")
                print(e)

        print(f"Success {success_count}/{total_count} envs")
        print(f"Expect {supervised_count} envs to support supervised learning")


def test_print_all():
    """Test printing of all experiments."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        for env_name in ENVS:
            print()
            print(f"Test printing env: {env_name}")
            env = ngym.make(env_name)
            print(env)


def _test_trialenv(env):
    """Test if a TrialEnv is behaving correctly."""
    if isinstance(env, str):
        env = ngym.make(env)
    elif not isinstance(env, gym.Env):
        msg = f"{type(env)=} must be a string or a gym.Env"
        raise TypeError(msg)

    trial = env.new_trial()
    assert (
        trial is not None
    ), f"TrialEnv should return trial info dict {env}"  # FIXME: should we assert isinstance(trial, dict) instead?


def test_trialenv_all():
    """Test if all environments can at least be run."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        for env_name in ENVS:
            env = ngym.make(env_name)
            if not isinstance(env, ngym.TrialEnv):  # FIXME: probably these should be flagged rather than skipped
                continue
            _test_trialenv(env)


def _test_seeding(env):
    """Test if environments are replicable."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        if env is None:
            env = ngym.all_envs()[0]

        if isinstance(env, str):
            kwargs = {"dt": 20}
            env = ngym.make(env, **kwargs)
        elif not isinstance(env, gym.Env):
            msg = f"{type(env)=} must be a string or a gym.Env"
            raise TypeError(msg)

        env.seed(SEED)
        env.reset()
        ob_mat = []
        rew_mat = []
        act_mat = []
        for _ in range(100):
            action = env.action_space.sample()
            ob, rew, terminated, _truncated, _info = env.step(action)
            ob_mat.append(ob)
            rew_mat.append(rew)
            act_mat.append(action)
            if terminated:
                env.reset()
        ob_mat = np.array(ob_mat)
        rew_mat = np.array(rew_mat)
        act_mat = np.array(act_mat)
        # FIXME: given the returns below, it seems as though this should be the helper function for the test below
        # rather than its own test, except that the first env is chosen seemingly arbitrarily. This can be done in next
        # fucntion instead to avoid the returns in an actual test. This should maybe be implemented for other tests here
        # as well
        return ob_mat, rew_mat, act_mat


# TODO: there is one env for which it sometimes raises an error
def test_seeding_all():
    """Test if all environments are replicable."""
    assert ngym.all_envs()[0] in ENVS_NOPSYCHOPY
    for env_name in sorted(ENVS_NOPSYCHOPY):
        print(f"Running env: {env_name}")
        obs1, rews1, acts1 = _test_seeding(env_name)
        obs2, rews2, acts2 = _test_seeding(env_name)
        assert (obs1 == obs2).all(), "obs are not identical"
        assert (rews1 == rews2).all(), "rewards are not identical"
        assert (acts1 == acts2).all(), "actions are not identical"


def test_plot_all():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        for env_name in ENVS:
            if env_name == "Null-v0":
                continue
            env = ngym.make(env_name, dt=20)
            action = np.zeros_like(env.action_space.sample())
            try:  # FIXME: no actual test is run, as errors are caught
                ngym.utils.plot_env(env, num_trials=2, def_act=action)
            except Exception as e:  # noqa: BLE001 # FIXME: unclear which error is expected here.
                print(f"Error in plotting env: {env_name}, {e}")
                print(e)
            plt.close()
