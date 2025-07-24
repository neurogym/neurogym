import warnings
from importlib.util import find_spec
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from matplotlib import pyplot as plt

import neurogym as ngym
from neurogym.core import BaseEnv, TrialEnv
from neurogym.envs.registration import all_envs, all_tags, make
from neurogym.utils.data import Dataset
from neurogym.utils.logging import logger
from tests import ANNUBES_KWS

_HAVE_PSYCHOPY = find_spec("psychopy") is not None  # check if psychopy is installed
SEED = 0

ENVS = all_envs(psychopy=_HAVE_PSYCHOPY, contrib=True, collections=True)
# Envs without psychopy, TODO: check if contrib or collections include psychopy
ENVS_NOPSYCHOPY = all_envs(psychopy=False, contrib=True, collections=True)


def _test_run(env_name: str, num_steps: int = 100) -> gym.Env:
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


def test_run_all() -> None:
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


def _test_dataset(env: str) -> None:
    """Main function for testing if an environment is healthy."""
    if env.startswith("Null"):
        return
    env_kwargs: dict[str, Any] = {"dt": 20}
    if env.startswith("Annubes"):
        env_kwargs.update(ANNUBES_KWS)
    dataset = Dataset(env, env_kwargs=env_kwargs, batch_size=16, seq_len=300, cache_len=10_000)
    for _ in range(10):
        inputs, target = dataset()
        assert inputs.shape[0] == target.shape[0]


def test_dataset_all() -> None:
    """Test if all environments can at least be run."""
    failing_envs = ("Bandit", "DawTwoStep", "EconomicDecisionMaking")
    # TODO: the envs above are failing this test and are skipped for the time being.
    # The issue seems to be related to setting the ground truth for these envs.

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Casting input x to numpy array.*")
        warnings.filterwarnings("ignore", message=".*is not within the observation space*")
        warnings.filterwarnings("ignore", message=".*method was expecting numpy array dtype to be*")
        warnings.filterwarnings("ignore", message=".*method was expecting a numpy array*")
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        for env_name in all_envs():
            if any(env_name.startswith(failing) for failing in failing_envs):
                continue
            try:
                _test_dataset(env_name)
            except Exception:
                logger.error(f"Failure at running env: {env_name}")
                raise


def _test_trialenv(env_name: str) -> None:
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


def test_trialenv_all() -> None:
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


def _test_seeding(env_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Test if environments are replicable."""
    env_kwargs: dict[str, Any] = {"dt": 20}
    if env_name.startswith("Annubes"):
        env_kwargs.update(ANNUBES_KWS)
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
def test_seeding_all() -> None:
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


def test_plot_all() -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*get variables from other wrappers is deprecated*")
        warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include `render_modes`*")
        for env_name in ENVS:
            if env_name.startswith("Null"):
                continue
            env_kwargs: dict[str, Any] = {"dt": 20}
            if env_name.startswith("Annubes"):
                env_kwargs.update(ANNUBES_KWS)
            env = make(env_name, **env_kwargs)
            action = np.zeros_like(env.action_space.sample())
            try:
                ngym.utils.plotting.plot_env(env, num_trials=2, def_act=action)
            except Exception as e:
                logger.error(f"Error in plotting env: {env_name}, {e}")
                plt.close()
                raise


def test_get_envs() -> None:
    for task in ["GoNogo-v0", "GoNogo"]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Using the latest versioned environment*")
            warnings.filterwarnings("ignore", message=".*The environment creator metadata doesn't include*")
            env = make(task)
        assert isinstance(env, gym.Env)
        assert env.spec.id == "GoNogo-v0"  # type: ignore[union-attr]
    for invalid in ["GoNogo-v99", "GoGoNo"]:
        with pytest.raises(gym.error.UnregisteredEnv):
            _env = make(invalid)
