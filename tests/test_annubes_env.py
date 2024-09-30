import numpy as np
import pytest

from neurogym.envs.annubes import AnnubesEnv

RND_SEED = 42


@pytest.fixture
def default_env():
    """Fixture for creating a default AnnubesEnv instance."""
    return AnnubesEnv()


@pytest.fixture
def custom_env():
    """Fixture for creating a custom AnnubesEnv instance with specific parameters."""
    return AnnubesEnv(
        session={"v": 1},
        stim_intensities=[0.5, 1.0],
        stim_time=800,
        catch_prob=0.3,
        fix_intensity=0.1,
        fix_time=300,
        dt=50,
        tau=80,
        output_behavior=[0, 0.5, 1],
        noise_std=0.02,
        rewards={"abort": -0.2, "correct": +1.5, "fail": -0.5},
        random_seed=42,
    )


@pytest.mark.parametrize(
    ("fix_time", "expected_type"),
    [
        (500, int),  # Fixed integer duration
        (500.0, float),  # Fixed float duration
        (lambda: 500, int),  # Callable returning fixed duration
        ([300, 400, 500], np.integer),  # List of durations for random choice
        (("uniform", (300, 500)), float),  # Uniform distribution
        (("choice", [300, 400, 500]), np.integer),  # Choice from list
        (("truncated_exponential", (300, 500, 400)), int),  # Truncated exponential
        (("constant", 500), int),  # Constant value
        (("until", 15000), int),  # Until specified time
    ],
)
def test_fix_time_types(fix_time, expected_type):
    """Test various types of fix_time specifications."""
    env = AnnubesEnv(fix_time=fix_time)
    # Be sure that at least one trial is run to sample the fix_time
    env.reset()

    # Check if the sampled fix_time is of the expected type
    assert isinstance(
        env._duration["fixation"],
        expected_type,
    ), f"Expected fix_time to be of type {expected_type}, but got {type(env._duration['fixation'])}"

    # Check if the sampled fix_time is in the given list of values
    if isinstance(fix_time, list):
        assert (
            env._duration["fixation"] in fix_time
        ), f"Expected fix_time to be one of {fix_time}, but got {env._duration['fixation']}"
    # Check if the sampled fix_time is in the given range
    elif isinstance(fix_time, tuple) and fix_time[0] == "uniform":
        fix_time_range = fix_time[1]
        assert (
            fix_time_range[0] <= env._duration["fixation"] <= fix_time_range[1]
        ), f"""Expected fix_time to be between {fix_time_range[0]} and {fix_time_range[1]},
        but got {env._duration['fixation']}"""
    # Check if the sampled fix_time is in the given list of values
    elif isinstance(fix_time, tuple) and fix_time[0] == "choice":
        assert (
            env._duration["fixation"] in fix_time[1]
        ), f"Expected fix_time to be one of {fix_time[1]}, but got {env._duration['fixation']}"
    # Check if the sampled fix_time is in the given range
    elif isinstance(fix_time, tuple) and fix_time[0] == "truncated_exponential":
        fix_time_range = fix_time[1]
        assert (
            fix_time_range[0] <= env._duration["fixation"] <= fix_time_range[1]
        ), f"""Expected fix_time to be between {fix_time_range[0]} and {fix_time_range[1]},
        but got {env._duration['fixation']}"""
    # Check if the sampled fix_time is the given constant value
    elif isinstance(fix_time, tuple) and fix_time[0] == "constant":
        assert (
            env._duration["fixation"] == fix_time[1]
        ), f"Expected fix_time to be {fix_time[1]}, but got {env._duration['fixation']}"

    # For callable fix_time, check if it's actually called
    if callable(fix_time):
        assert env._duration["fixation"] == 500, f"Expected fix_time to be 500, but got {env._duration['fixation']}"

    # Ensure the sampled time is a multiple of dt
    assert (
        env._duration["fixation"] % env.dt == 0
    ), f"Expected fix_time to be a multiple of dt ({env.dt}), but got {env._duration['fixation']}"


def test_observation_space(default_env, custom_env):
    """Test the observation space of both default and custom environments.

    This test checks:
    1. The shape of the observation space
    2. The names assigned to each dimension of the observation space
    """
    assert default_env.observation_space.shape == (4,)
    assert custom_env.observation_space.shape == (3,)

    assert default_env.observation_space.name == {"fixation": 0, "start": 1, "v": 2, "a": 3}
    assert custom_env.observation_space.name == {"fixation": 0, "start": 1, "v": 2}


def test_action_space(default_env, custom_env):
    """Test the action space of both default and custom environments.

    This test checks:
    1. The number of possible actions
    2. The names and values assigned to each action
    """
    assert default_env.action_space.n == 2
    assert custom_env.action_space.n == 3

    assert default_env.action_space.name == {"fixation": 0, "choice": [1]}
    assert custom_env.action_space.name == {"fixation": 0, "choice": [0.5, 1]}


@pytest.mark.parametrize("env", ["default_env", "custom_env"])
def test_step(request, env):
    """Test the step function of the environment.

    This test checks:
    1. Correct and incorrect actions during fixation period
    2. Correct and incorrect actions during stimulus period
    3. Rewards given for different actions
    4. Termination conditions
    """
    # Get the environment fixture
    env = request.getfixturevalue(env)
    env.reset()

    # Test fixation period
    _, reward, terminated, truncated, _ = env.step(0)  # Correct fixation
    assert not terminated
    assert not truncated
    assert reward == 0

    _, reward, terminated, truncated, _ = env.step(1)  # Incorrect fixation
    assert not terminated
    assert not truncated
    assert reward == env.rewards["abort"]

    # Move to stimulus period
    while env.in_period("fixation"):
        env.step(0)

    # Test stimulus period
    _, reward, terminated, truncated, _ = env.step(env.gt_now)  # Correct choice
    assert not terminated
    assert not truncated
    assert reward == env.rewards["correct"]

    env.reset()
    while env.in_period("fixation"):
        env.step(0)

    _, reward, terminated, truncated, _ = env.step(
        (env.gt_now + 1) % env.action_space.n,
    )  # Incorrect choice
    assert not terminated
    assert not truncated
    assert reward == env.rewards["fail"]


def test_initial_state_and_first_reset(default_env):
    """Test the initial state of the environment and its state after the first reset.

    This test checks:
    1. Initial values of time step and trial number
    2. Values of time step and trial number after first reset
    3. Shape of the first observation
    4. Presence of trial information after reset
    """
    # Check initial state
    assert default_env.t == 0, f"t={default_env.t}, should be 0 initially"
    assert default_env.num_tr == 0, f"num_tr={default_env.num_tr}, should be 0 initially"

    # Check state after first reset
    ob, _ = default_env.reset()
    assert default_env.t == 100, f"default_env={default_env.t}, should be 100 after first reset"
    assert default_env.num_tr == 1, f"num_tr={default_env.num_tr}, should be 1 after first reset"
    assert isinstance(default_env.trial, dict)
    assert (
        ob.shape == default_env.observation_space.shape
    ), f"observation_space.shape={default_env.observation_space.shape}, should match the observation space"


def test_random_seed_reproducibility():
    """Test the reproducibility of the environment when using the same random seed.

    This test checks if two environments initialized with the same seed produce identical trials.
    """
    for _ in range(10):
        env1 = AnnubesEnv(random_seed=RND_SEED)
        env2 = AnnubesEnv(random_seed=RND_SEED)
        trial1 = env1._new_trial()
        trial2 = env2._new_trial()
        assert trial1 == trial2


@pytest.mark.parametrize("catch_prob", [0.0, 0.3, 0.7, 1.0])
def test_catch_prob(catch_prob):
    """Test if the catch trial probability is working as expected.

    This test checks:
    1. If the observed probability of catch trials matches the specified probability
    2. If the probability works correctly for various values including edge cases (0.0 and 1.0)
    """
    n_trials = 1000
    env = AnnubesEnv(catch_prob=catch_prob, random_seed=RND_SEED)
    catch_count = 0

    for _ in range(n_trials):
        env.reset()
        trial_info = env.trial
        if trial_info["catch"]:
            catch_count += 1

    observed_prob = catch_count / n_trials
    expected_prob = catch_prob

    # Check if the observed probability is close to the expected probability
    assert np.isclose(
        observed_prob,
        expected_prob,
        atol=0.05,
    ), f"Expected catch probability {expected_prob}, but got {observed_prob}"
