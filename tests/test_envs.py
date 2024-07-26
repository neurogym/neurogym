import warnings

import gymnasium as gym
import numpy as np

import neurogym as ngym
from neurogym.utils.data import Dataset

try:
    import psychopy  # noqa: F401

    _have_psychopy = True
except ImportError:
    _have_psychopy = False

ENVS = ngym.all_envs(psychopy=_have_psychopy, contrib=True, collections=True)
# Envs without psychopy, TODO: check if contrib or collections include psychopy
ENVS_NOPSYCHOPY = ngym.all_envs(psychopy=False, contrib=True, collections=True)


def make_env(env, **kwargs):
    # use ngym.make and not gym.make to disable env_checker
    return ngym.make(env, **kwargs)


def test_run(env=None, num_steps=100, verbose=False, **kwargs):
    """Test if one environment can at least be run."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        if env is None:
            env = ngym.all_envs()[0]
        if isinstance(env, str):
            env = make_env(env, **kwargs)
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
        for env_name in sorted(ENVS):
            print(env_name)
            test_run(env_name, verbose=verbose_success)


def test_dataset(env=None):
    """Main function for testing if an environment is healthy."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        if env is None:
            env = ngym.all_envs()[0]

        print("Testing Environment:", env)
        kwargs = {"dt": 20}
        dataset = Dataset(env, env_kwargs=kwargs, batch_size=16, seq_len=300, cache_len=1e4)
        for _ in range(10):
            inputs, target = dataset()
            assert inputs.shape[0] == target.shape[0]


def test_dataset_all():
    """Test if all environments can at least be run."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        success_count = 0
        total_count = len(ngym.all_envs())
        supervised_count = len(ngym.all_envs(tag="supervised"))
        for env_name in sorted(ngym.all_envs()):
            print(f"Running env: {env_name}")
            try:
                test_dataset(env_name)
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
        for env_name in sorted(ENVS):
            print()
            print(f"Test printing env: {env_name}")
            env = make_env(env_name)
            print(env)


def test_trialenv(env=None, **kwargs):
    """Test if a TrialEnv is behaving correctly."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        if env is None:
            env = ngym.all_envs()[0]
        if isinstance(env, str):
            env = make_env(env, **kwargs)
        elif not isinstance(env, gym.Env):
            msg = f"{type(env)=} must be a string or a gym.Env"
            raise TypeError(msg)
        trial = env.new_trial()
        assert trial is not None, f"TrialEnv should return trial info dict {env}"


def test_trialenv_all():
    """Test if all environments can at least be run."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        for env_name in sorted(ENVS):
            env = make_env(env_name)
            if not isinstance(env, ngym.TrialEnv):
                continue
            test_trialenv(env)


def test_seeding(env=None, seed=0):
    """Test if environments are replicable."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        if env is None:
            env = ngym.all_envs()[0]

        if isinstance(env, str):
            kwargs = {"dt": 20}
            env = make_env(env, **kwargs)
        elif not isinstance(env, gym.Env):
            msg = f"{type(env)=} must be a string or a gym.Env"
            raise TypeError(msg)

        env.seed(seed)
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
        return ob_mat, rew_mat, act_mat


# TODO: there is one env for which it sometimes raises an error
def test_seeding_all():
    """Test if all environments are replicable."""
    for env_name in sorted(ENVS_NOPSYCHOPY):
        print(f"Running env: {env_name}")
        obs1, rews1, acts1 = test_seeding(env_name, seed=0)
        obs2, rews2, acts2 = test_seeding(env_name, seed=0)
        assert (obs1 == obs2).all(), "obs are not identical"
        assert (rews1 == rews2).all(), "rewards are not identical"
        assert (acts1 == acts2).all(), "actions are not identical"


def test_plot_envs():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        for env_name in sorted(ENVS):
            if env_name == "Null-v0":
                continue
            env = make_env(env_name, dt=20)
            action = np.zeros_like(env.action_space.sample())
            try:
                ngym.utils.plot_env(env, num_trials=2, def_act=action)
            except Exception as e:  # noqa: BLE001 # FIXME: unclear which error is expected here.
                print(f"Error in plotting env: {env_name}, {e}")
                print(e)
