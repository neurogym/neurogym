import warnings
from importlib.util import find_spec

import gymnasium as gym
import numpy as np
import pytest
from matplotlib import pyplot as plt

import neurogym as ngym
from neurogym.core import BaseEnv, TrialEnv
from neurogym.envs.registration import all_envs, all_tags, make
from neurogym.utils.data import Dataset
from neurogym.utils.logging import logger

from . import ANNUBES_KWS

_HAVE_PSYCHOPY = find_spec("psychopy") is not None  # check if psychopy is installed
SEED = 0

ENVS = all_envs(psychopy=_HAVE_PSYCHOPY, contrib=True, collections=True)
# Envs without psychopy, TODO: check if contrib or collections include psychopy
ENVS_NOPSYCHOPY = all_envs(psychopy=False, contrib=True, collections=True)


def _test_run(env_name: str, num_steps: int = 100):
    """Test if one environment can at least be run."""
    env_kwargs = {}
    if env_name.startswith("Annubes"):
        env_kwargs = ANNUBES_KWS
    env = make(env_name, **env_kwargs)

    env.reset()
    for _ in range(num_steps):
        action = env.action_space.sample()
        _state, _rew, terminated, _truncated, _info = env.step(action)
        if terminated:
            env.reset()

    tags = env.metadata.get("tags", [])
    assert all(t in all_tags() for t in tags)

    return env


def test_run_all():
    """Test if all environments can at least be run."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        try:
            for env_name in ENVS:
                _test_run(env_name)
        except:
            logger.error(f"Failure at running env: {env_name}")
            raise


def _test_dataset(env: str):
    """Main function for testing if an environment is healthy."""
    kwargs = {"dt": 20}
    dataset = Dataset(env, env_kwargs=kwargs, batch_size=16, seq_len=300, cache_len=10_000)
    for _ in range(10):
        inputs, target = dataset()
        assert inputs.shape[0] == target.shape[0]


@pytest.mark.skip(reason="This test is not actually performed, as any error is caught away.")
def test_dataset_all():
    """Test if all environments can at least be run."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Casting input x to numpy array.*")
        warnings.filterwarnings("ignore", message=".*is not within the observation space*")
        warnings.filterwarnings("ignore", message=".*method was expecting numpy array dtype to be*")
        warnings.filterwarnings("ignore", message=".*method was expecting a numpy array*")
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        success_count = 0
        total_count = len(all_envs())
        supervised_count = len(all_envs(tag="supervised"))
        for env_name in all_envs():
            try:  # FIXME, tests are not actually performed here, as any error is caught away
                _test_dataset(env_name)
                success_count += 1
            except Exception as e:  # noqa: PERF203, BLE001 # FIXME: unclear which error is expected here.
                logger.error(f"Failure at running env: {env_name}")
                logger.error(e)

        if success_count < total_count:
            logger.info(f"Success {success_count}/{total_count} envs")
        logger.debug(f"Expect {supervised_count} envs to support supervised learning")


def _test_trialenv(env_name: str):
    """Test if a TrialEnv is behaving correctly."""
    env_kwargs = {}
    if env_name.startswith("Annubes"):
        env_kwargs = ANNUBES_KWS
    env = make(env_name, **env_kwargs)

    if isinstance(env.unwrapped, TrialEnv):
        trial = env.new_trial()  # type: ignore[attr-defined]
        assert isinstance(trial, dict)
    else:
        msg = f"Environment {env_name} is not a subclass of `TrialEnv`."
        raise TypeError(msg)


def test_trialenv_all():
    """Test if all environments can at least be run."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        try:
            for env_name in ENVS:
                _test_trialenv(env_name)
        except:
            logger.error(f"Failure with env: {env_name}")
            raise


def _test_seeding(env_name: str):
    """Test if environments are replicable."""
    env_kwargs = {"dt": 20}
    if env_name.startswith("Annubes"):
        env_kwargs.update(ANNUBES_KWS)  # type: ignore[arg-type]
        env_kwargs.update({"random_seed": SEED})
    env = make(env_name, **env_kwargs)

    if isinstance(env.unwrapped, BaseEnv):
        env.seed(SEED)  # type: ignore[attr-defined]
    else:
        msg = f"Environment {env_name} is not a subclass of `BaseEnv`."
        raise TypeError(msg)
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
    return np.array(ob_mat), np.array(rew_mat), np.array(act_mat)


# TODO: there is one env for which it sometimes raises an error
def test_seeding_all():
    """Test if all environments are replicable."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        try:
            for env_name in sorted(ENVS_NOPSYCHOPY):
                obs1, rews1, acts1 = _test_seeding(env_name)
                obs2, rews2, acts2 = _test_seeding(env_name)
                assert (obs1 == obs2).all(), "obs are not identical"
                assert (rews1 == rews2).all(), "rewards are not identical"
                assert (acts1 == acts2).all(), "actions are not identical"
        except:
            logger.error(f"Failure with env: {env_name}")
            raise


@pytest.mark.skip(reason="This test is not actually performed, as any error is caught away.")
def test_plot_all():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        for env_name in ENVS:
            env_kwargs = {"dt": 20}
            if env_name.startswith("Null"):
                continue
            if env_name.startswith("Annubes"):
                env_kwargs.update(ANNUBES_KWS)
            env = make(env_name, **env_kwargs)
            action = np.zeros_like(env.action_space.sample())
            try:  # FIXME: no actual test is run, as errors are caught
                ngym.utils.plotting.plot_env(env, num_trials=2, def_act=action)
            except Exception as e:  # noqa: BLE001 # FIXME: unclear which error is expected here.
                logger.error(f"Error in plotting env: {env_name}, {e}")
                logger.error(e)
            plt.close()


def test_get_envs():
    for task in ["GoNogo-v0", "GoNogo"]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Using the latest versioned environment*")
            warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include*")
            env = make(task)
        assert isinstance(env, gym.Env)
        assert env.spec.id == "GoNogo-v0"
    for invalid in ["GoNogo-v99", "GoGoNo"]:
        with pytest.raises(gym.error.UnregisteredEnv):
            _env = make(invalid)
