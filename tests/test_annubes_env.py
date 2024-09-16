import numpy as np
import pytest

from neurogym.envs.annubes import AnnubesEnv

RND_SEED = 42


@pytest.fixture
def default_env():
    return AnnubesEnv()


@pytest.fixture
def custom_env():
    return AnnubesEnv(
        session={"v": 1},
        stim_intensities=[0.5, 1.0],
        stim_time=800,
        catch_prob=0.3,
        fix_intensity=0.1,
        fix_time=300,
        dt=50,
        tau=80,
        n_outputs=3,
        output_behavior=[0, 0.5, 1],
        noise_std=0.02,
        rewards={"abort": -0.2, "correct": +1.5, "fail": -0.5},
        random_seed=42,
    )


def test_observation_space(default_env, custom_env):
    assert default_env.observation_space.shape == (4,)
    assert custom_env.observation_space.shape == (3,)

    assert default_env.observation_space.name == {"fixation": 0, "start": 1, "v": 2, "a": 3}
    assert custom_env.observation_space.name == {"fixation": 0, "start": 1, "v": 2}


def test_action_space(default_env, custom_env):
    assert default_env.action_space.n == 2
    assert custom_env.action_space.n == 3

    assert default_env.action_space.name == {"fixation": 0, "choice": [0, 1]}
    assert custom_env.action_space.name == {"fixation": 0, "choice": [0, 0.5, 1]}


@pytest.mark.parametrize("env", ["default_env", "custom_env"])
def test_step(request, env):
    # Get the environment fixture
    env = request.getfixturevalue(env)
    env.reset()

    # Test fixation period
    ob, reward, terminated, truncated, info = env.step(0)  # Correct fixation
    assert not terminated
    assert not truncated
    assert reward == 0

    ob, reward, terminated, truncated, info = env.step(1)  # Incorrect fixation
    assert not terminated
    assert not truncated
    assert reward == env.rewards["abort"]

    # Move to stimulus period
    while env.in_period("fixation"):
        env.step(0)

    # Test stimulus period
    ob, reward, terminated, truncated, info = env.step(env.gt_now)  # Correct choice
    assert not terminated
    assert not truncated
    assert reward == env.rewards["correct"]

    env.reset()
    while env.in_period("fixation"):
        env.step(0)

    ob, reward, terminated, truncated, info = env.step(
        (env.gt_now + 1) % env.action_space.n,
    )  # Incorrect choice
    assert not terminated
    assert not truncated
    assert reward == env.rewards["fail"]


def test_initial_state_and_first_reset(default_env):
    # Check initial state
    assert default_env.t == 0, "t should be 0 initially"
    assert default_env.num_tr == 0, "num_tr should be 0 initially"

    # Check state after first reset
    ob, _ = default_env.reset()
    assert default_env.t == 100, "t should be 100 after first reset"
    assert default_env.num_tr == 1, "num_tr should be 1 after first reset"
    assert isinstance(default_env.trial, dict)
    assert ob.shape == default_env.observation_space.shape, "Observation shape should match the observation space"


def test_random_seed_reproducibility():
    for _ in range(10):
        env1 = AnnubesEnv(random_seed=42)
        env2 = AnnubesEnv(random_seed=42)
        trial1 = env1._new_trial()
        trial2 = env2._new_trial()
        assert trial1 == trial2


@pytest.mark.parametrize("catch_prob", [0.0, 0.3, 0.7, 1.0])
def test_catch_prob(catch_prob):
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
