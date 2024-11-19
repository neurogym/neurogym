from collections.abc import Callable

import numpy as np
import pytest

from neurogym.envs.annubes import AnnubesEnv

RND_SEED = 42
FIX_INTENSITY = 0.1
N_TRIALS = 1000
OUTPUT_BEHAVIOR = [0, 0.5, 1]


@pytest.fixture
def default_env() -> AnnubesEnv:
    """Fixture for creating a default AnnubesEnv instance."""
    return AnnubesEnv()


@pytest.fixture
def custom_env() -> AnnubesEnv:
    """Fixture for creating a custom AnnubesEnv instance with specific parameters."""
    return AnnubesEnv(
        session={"v": 1},
        stim_intensities=[0.5, 1.0],
        stim_time=800,
        catch_prob=0.3,
        fix_intensity=FIX_INTENSITY,
        fix_time=300,
        dt=50,
        tau=80,
        output_behavior=OUTPUT_BEHAVIOR,
        noise_std=0.02,
        rewards={"abort": -0.2, "correct": +1.5, "fail": -0.5},
        random_seed=42,
    )


@pytest.mark.parametrize(
    ("time", "expected_type"),
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
def test_fix_time_types(
    time: float | Callable | list[int] | tuple[str, tuple[int, int] | list[int]],
    expected_type: type,
) -> None:
    """Test various types of fix_time specifications."""
    env = AnnubesEnv(fix_time=time, iti=time)
    # Be sure that at least one trial is run to sample the fix_time
    env.reset()

    # Check if the sampled fix_time is of the expected type
    for t in ["fixation", "iti"]:
        assert isinstance(
            env._duration[t],
            expected_type,
        ), f"Expected {t} time to be of type {expected_type}, but got {type(env._duration['fixation'])}"

        # Check if the sampled time is in the given list of values
        if isinstance(time, list):
            assert (
                env._duration[t] in time
            ), f"Expected {t} time to be one of {time}, but got {env._duration['fixation']}"
        # Check if the sampled time is in the given range
        elif isinstance(time, tuple) and time[0] == "uniform":
            time_range = time[1]
            assert (
                time_range[0] <= env._duration[t] <= time_range[1]
            ), f"""Expected {t} time to be between {time_range[0]} and {time_range[1]},
            but got {env._duration['fixation']}"""
        # Check if the sampled time is in the given list of values
        elif isinstance(time, tuple) and time[0] == "choice":
            assert (
                env._duration[t] in time[1]
            ), f"Expected {t} time to be one of {time[1]}, but got {env._duration['fixation']}"
        # Check if the sampled time is in the given range
        elif isinstance(time, tuple) and time[0] == "truncated_exponential":
            time_range = time[1]
            assert (
                time_range[0] <= env._duration[t] <= time_range[1]
            ), f"""Expected {t} time to be between {time_range[0]} and {time_range[1]},
            but got {env._duration['fixation']}"""
        # Check if the sampled time is the given constant value
        elif isinstance(time, tuple) and time[0] == "constant":
            assert (
                env._duration[t] == time[1]
            ), f"Expected {t} time to be {time[1]}, but got {env._duration['fixation']}"

        # For callable time, check if it's actually called
        if callable(time):
            assert env._duration[t] == 500, f"Expected {t} time to be 500, but got {env._duration['fixation']}"

        # Ensure the sampled time is a multiple of dt
        assert (
            env._duration[t] % env.dt == 0
        ), f"Expected {t} time to be a multiple of dt ({env.dt}), but got {env._duration['fixation']}"


@pytest.mark.parametrize("catch_prob", [0.0, 0.3, 0.7, 1.0])
def test_catch_prob(catch_prob: float) -> None:
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


@pytest.mark.parametrize(
    ("session", "catch_prob", "max_sequential"),
    [
        ({"v": 0.5, "a": 0.5}, 0.5, 3),
        ({"v": 0.3, "a": 0.7}, 0.1, 4),
        ({"v": 0.3, "a": 0.3, "av": 0.3}, 0.7, 4),
    ],
)
def test_annubes_env_max_sequential(session: dict, catch_prob: float, max_sequential: int) -> None:
    """Test the maximum sequential trial constraint in the AnnubesEnv.

    The test performs the following checks:
    1. Ensures no sequence of non-catch trials longer than `max_sequential`
       of the same type occurs.
    2. Verifies that all specified trial types in the session occur.
    3. Checks if the distribution of trial types is roughly balanced
       according to the specified probabilities.
    4. Verifies that the frequency of catch trials matches the specified
       `catch_prob`.
    """
    env = AnnubesEnv(session=session, catch_prob=catch_prob, max_sequential=max_sequential, random_seed=RND_SEED)

    trial_types: list[str | None] = []
    for _ in range(N_TRIALS):
        env.new_trial()
        trial_types.append(env.trial["stim_type"])

    # Check for sequences longer than max_sequential, excluding None (catch trials)
    for i in range(len(trial_types) - max_sequential):
        sequence = [t for t in trial_types[i : i + max_sequential + 1] if t is not None]
        if len(sequence) > max_sequential:
            assert len(set(sequence)) > 1, f"Found a sequence longer than {max_sequential} at index {i}"

    # Check that all the trial types occur
    assert set(trial_types) - {None} == set(env.session.keys()), "Not all trial types occurred"

    # Check that the distribution is roughly balanced
    expected_non_catch = N_TRIALS * (1 - catch_prob)

    for trial_type, expected_prob in session.items():
        count = trial_types.count(trial_type)
        expected_count = expected_non_catch * expected_prob
        # Allow for 20% deviation from expected count
        assert (
            0.7 * expected_count <= count <= 1.3 * expected_count
        ), f"{trial_type} trials are not balanced. Expected: {expected_count:.2f}, Actual: {count}"

    # Check catch trial frequency
    catch_count = trial_types.count(None)
    expected_catch = N_TRIALS * catch_prob
    assert (
        0.7 * expected_catch <= catch_count <= 1.3 * expected_catch
    ), f"Catch trials are not balanced. Expected: {expected_catch:.2f}, Actual: {catch_count}"


def test_observation_space(default_env: AnnubesEnv, custom_env: AnnubesEnv) -> None:
    """Test the observation space of both default and custom environments.

    This test checks:
    1. The shape of the observation space
    2. The names assigned to each dimension of the observation space
    """
    assert default_env.observation_space.shape == (4,)
    assert custom_env.observation_space.shape == (3,)

    assert default_env.observation_space.name == {"fixation": 0, "start": 1, "v": 2, "a": 3}
    assert custom_env.observation_space.name == {"fixation": 0, "start": 1, "v": 2}


def test_action_space(default_env: AnnubesEnv, custom_env: AnnubesEnv) -> None:
    """Test the action space of both default and custom environments.

    This test checks:
    1. The number of possible actions
    2. The names and values assigned to each action
    """
    assert default_env.action_space.n == 2
    assert custom_env.action_space.n == len(OUTPUT_BEHAVIOR)

    assert default_env.action_space.name == {"fixation": 0, "choice": [1]}
    assert custom_env.action_space.name == {"fixation": FIX_INTENSITY, "choice": OUTPUT_BEHAVIOR[1:]}


@pytest.mark.parametrize("env", ["default_env", "custom_env"])
def test_step(request, env: str) -> None:
    """Test the step function of the environment.

    This test checks:
    1. Correct and incorrect actions during fixation period
    2. Correct and incorrect actions during stimulus period
    3. Rewards given for different actions
    4. Termination conditions
    """
    # Get the environment fixture
    env_test = request.getfixturevalue(env)
    env_test.reset()

    # Test fixation period
    _, reward, terminated, truncated, _ = env_test.step(0)  # Correct fixation
    assert not terminated
    assert not truncated
    assert reward == 0

    _, reward, terminated, truncated, _ = env_test.step(1)  # Incorrect fixation
    assert not terminated
    assert not truncated
    assert reward == env_test.rewards["abort"]

    # Move to stimulus period
    while env_test.in_period("fixation"):
        env_test.step(0)

    # Test stimulus period
    _, reward, terminated, truncated, _ = env_test.step(env_test.gt_now)  # Correct choice
    assert not terminated
    assert not truncated
    assert reward == env_test.rewards["correct"]

    env_test.reset()
    while env_test.in_period("fixation"):
        env_test.step(0)

    _, reward, terminated, truncated, _ = env_test.step(
        (env_test.gt_now + 1) % env_test.action_space.n,
    )  # Incorrect choice
    assert not terminated
    assert not truncated
    assert reward == env_test.rewards["fail"]


def test_initial_state_and_first_reset(default_env: AnnubesEnv) -> None:
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


def test_random_seed_reproducibility() -> None:
    """Test the reproducibility of the environment when using the same random seed.

    This test checks if two environments initialized with the same seed produce identical trials.
    """
    for _ in range(10):
        env1 = AnnubesEnv(random_seed=RND_SEED)
        env2 = AnnubesEnv(random_seed=RND_SEED)
        trial1 = env1._new_trial()
        trial2 = env2._new_trial()
        assert trial1 == trial2
