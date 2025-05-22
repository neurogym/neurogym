from collections.abc import Callable
from itertools import combinations

import numpy as np
import pytest

from neurogym.envs.native.annubes import AnnubesEnv

RND_SEED = 42
FIX_INTENSITY = 0.1
N_TRIALS = 2500
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
        stim_intensities={"v": [0.5, 1.0]},
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
            assert env._duration[t] in time, (
                f"Expected {t} time to be one of {time}, but got {env._duration['fixation']}"
            )
        # Check if the sampled time is in the given range
        elif isinstance(time, tuple) and time[0] == "uniform":
            time_range = time[1]
            assert (
                time_range[0] <= env._duration[t] <= time_range[1]
            ), f"""Expected {t} time to be between {time_range[0]} and {time_range[1]},
            but got {env._duration["fixation"]}"""
        # Check if the sampled time is in the given list of values
        elif isinstance(time, tuple) and time[0] == "choice":
            assert env._duration[t] in time[1], (
                f"Expected {t} time to be one of {time[1]}, but got {env._duration['fixation']}"
            )
        # Check if the sampled time is in the given range
        elif isinstance(time, tuple) and time[0] == "truncated_exponential":
            time_range = time[1]
            assert (
                time_range[0] <= env._duration[t] <= time_range[1]
            ), f"""Expected {t} time to be between {time_range[0]} and {time_range[1]},
            but got {env._duration["fixation"]}"""
        # Check if the sampled time is the given constant value
        elif isinstance(time, tuple) and time[0] == "constant":
            assert env._duration[t] == time[1], (
                f"Expected {t} time to be {time[1]}, but got {env._duration['fixation']}"
            )

        # For callable time, check if it's actually called
        if callable(time):
            assert env._duration[t] == 500, f"Expected {t} time to be 500, but got {env._duration['fixation']}"

        # Ensure the sampled time is a multiple of dt
        assert env._duration[t] % env.dt == 0, (
            f"Expected {t} time to be a multiple of dt ({env.dt}), but got {env._duration['fixation']}"
        )


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
        if trial_info is not None and trial_info.get("catch"):
            catch_count += 1

    observed_prob = catch_count / n_trials
    expected_prob = catch_prob

    # Check if the observed probability is close to the expected probability
    assert np.isclose(
        observed_prob,
        expected_prob,
        atol=0.05,
    ), f"Expected catch probability {expected_prob}, but got {observed_prob}"


# TODO
# Deal with cases where the max_sequential value is too low.
# This can affect the probability distribution.
# One option might be to start with a higher tolerance (final assertion)
# and gradually tighten it as `max_sequential` increases.
@pytest.mark.parametrize("max_sequential", [5, 10, 15])
@pytest.mark.parametrize("exclusive", [True, False])
@pytest.mark.parametrize("catch_prob", np.linspace(0.0, 0.5, 5))
@pytest.mark.parametrize(
    "session",
    [
        # Modalities with equal probabilities summing up to 1
        {"v": 0.5, "a": 0.5},
        # Modalities with non-equal probabilities summing up to 1
        {"v": 0.3, "a": 0.7},
        # Modalities with equal probabilities with a sum < 1
        {"v": 0.3, "a": 0.3, "o": 0.3},
        # Modalities with equal probabilities with a sum > 1
        {"v": 0.3, "a": 0.7, "o": 0.9},
        # One modality with probability = 1
        {"v": 1, "a": 0.5, "o": 0.9},
        # One modality with probability = 0
        {"v": 0.0, "a": 0.5, "o": 0.9},
        # All modalities with probability = 1
        {"v": 1.0, "a": 1.0, "o": 1.0},
        # All modalities with probability = 0 (should fail)
        {"v": 0.0, "a": 0.0, "o": 0.0},
    ],
)
def test_annubes_env_max_sequential(
    session: dict,
    catch_prob: float,
    max_sequential: int,
    exclusive: bool,
):
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
    # Ensure that the environment fails to instantiate if all probability weights are 0
    if sum(session.values()) == 0.0:
        with pytest.raises(ValueError) as e:
            env = AnnubesEnv(
                session=session,
                catch_prob=catch_prob,
                max_sequential=max_sequential,
                exclusive=exclusive,
                random_seed=RND_SEED,
            )
        assert str(e.value) == "Please ensure that at least one modality has a non-zero probability."
        return

    env = AnnubesEnv(
        session=session,
        catch_prob=catch_prob,
        max_sequential=max_sequential,
        exclusive=exclusive,
        random_seed=RND_SEED,
    )

    # Prepare some useful variables
    # ==================================================
    # The set of all session modalities that have a probability > 0.0
    session_mods = {k for k, v in session.items() if v > 0.0}

    # The set of all possible combinations of the above, up to and
    # including the set of all modalities occurring together.
    mod_combinations = set()
    # Create all combinations of all possible lengths of all modalities
    for clen in range(1, len(session_mods) + 1):
        _combs = combinations(session_mods, clen)
        for c in _combs:
            mod_combinations.add(c[0] if len(c) == 1 else tuple(sorted(c)))

    # Collect rollouts
    # ==================================================
    # A dictionary of all occurrences of all combinations
    # that are expected to show up in the rollouts.
    occurrences = dict.fromkeys(mod_combinations, 0.0)
    occurrences[None] = 0.0

    # Sequential counts
    sequential_counts: dict = {k: [] for k in occurrences}
    for _ in range(N_TRIALS):
        trial = env.new_trial()
        stim_types = (None,) if trial["catch"] else tuple(sorted(trial["stim_types"]))

        # Unpack tuples with a single element.
        occurrences[stim_types[0] if len(stim_types) == 1 else stim_types] += 1.0

        # Store sequential occurrences, taking multi-sensory tasks into account
        for stim_type, seq_count in trial["sequential_count"].items():
            sequential_counts[stim_type].append(seq_count)

    # Make the occurrences relative
    occurrences = {k: v / N_TRIALS for k, v in occurrences.items()}

    # Check for sequences longer than max_sequential,
    # excluding catch trials.
    # ==================================================
    # First, turn all the sequences into NumPy arrays and remove the initial 0.
    # Also, max_sequential = 0 is interpreted as if there is no limit,
    # so we can use a very large number instead.
    if isinstance(max_sequential, int) and max_sequential == 0:
        max_sequential = np.iinfo(np.int32).max
    sequential_counts = {k: np.array(v[1:]) for k, v in sequential_counts.items()}
    for sequences in sequential_counts.values():
        # Get the indices of trials where the stimulus
        # occurred more than max_sequential times in a row.
        error_idx = np.argwhere(np.array(sequences, dtype=np.uint32) > max_sequential)
        assert len(error_idx) == 0, f"Found a sequence longer than {max_sequential} at trials {error_idx}"

    # Check that all the modalities occur,
    # unless the probability is 0.
    # ==================================================
    assert set(occurrences) - {None} == mod_combinations, "Not all modalities appeared in the trials"

    # Check that the distribution is roughly balanced.
    # We need to account for cases where multiple sensory modalities
    # are presented in parallel. In that case, the occurrences will
    # likely *not* sum up to 1.
    # Therefore, we check that the *relative* probabilities of all
    # modalities check out.
    # We add the catch probabilities separately.
    # ==================================================
    # Sum all the probabilities, excluding the catch probability
    session_sum = sum(session.values())
    session_rel_prob = np.array([session[k] / session_sum for k in session] + [catch_prob])

    # Relative probabilities for the session variables
    actual = dict.fromkeys(session, 0.0)
    for k in session:
        for ks, v in occurrences.items():
            if ks is None:
                continue
            if k == ks or k in ks:
                actual[k] += v
    actual_sum = sum(actual.values())

    # The actual probabilities as computed from the rollouts.
    actual_rel_prob = np.array([actual[k] / actual_sum for k in actual] + [occurrences[None]])

    # Ensure that the corresponding probabilities in the two arrays are within 5% of each other.
    assert np.allclose(session_rel_prob, actual_rel_prob, atol=5e-2, rtol=5e-2)


def test_observation_space(default_env: AnnubesEnv, custom_env: AnnubesEnv) -> None:
    """Test the observation space of both default and custom environments.

    This test checks:
    1. The shape of the observation space
    2. The names assigned to each dimension of the observation space
    """
    assert default_env.observation_space.shape == (4,)
    assert custom_env.observation_space.shape == (3,)

    assert default_env.observation_space.name == {"fixation": 0, "start": 1, "v": 2, "a": 3}  # type: ignore[attr-defined]
    assert custom_env.observation_space.name == {"fixation": 0, "start": 1, "v": 2}  # type: ignore[attr-defined]


def test_action_space(default_env: AnnubesEnv, custom_env: AnnubesEnv) -> None:
    """Test the action space of both default and custom environments.

    This test checks:
    1. The number of possible actions
    2. The names and values assigned to each action
    """
    assert default_env.action_space.n == 2  # type: ignore[attr-defined]
    assert custom_env.action_space.n == len(OUTPUT_BEHAVIOR)  # type: ignore[attr-defined]

    assert default_env.action_space.name == {"fixation": 0, "choice": [1]}  # type: ignore[attr-defined]
    assert custom_env.action_space.name == {"fixation": FIX_INTENSITY, "choice": OUTPUT_BEHAVIOR[1:]}  # type: ignore[attr-defined]


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
    assert ob.shape == default_env.observation_space.shape, (
        f"observation_space.shape={default_env.observation_space.shape}, should match the observation space"
    )


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


if __name__ == "__main__":
    test_annubes_env_max_sequential({"v": 0.5, "a": 0.8}, 0.5, 3)
