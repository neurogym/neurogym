from collections.abc import Callable

import numpy as np
import pytest

from neurogym.envs.native.annubes import CATCH_KEYWORD, CATCH_SESSION_KEY, AnnubesEnv

RND_SEED = 42
FIX_INTENSITY = 0.1
N_TRIALS = 1000
OUTPUT_BEHAVIOR = [0, 0.5, 1]


def _collect_trial_stats(env: AnnubesEnv, trials: int) -> tuple[dict, dict]:
    """Collect stimulus statistics over a number of trials.

    Args:
        env: An AnnubesEnv instance.
        trials: The size of the trial sample.

    Returns:
        - A dictionary mapping stimuli to their respective relative occurrences.
        - A dictionary mapping modalities and their sequential counts.
    """
    # Occurrences and sequential counts
    occurrences: dict = {}
    sequential_counts: dict = {}

    for _ in range(trials):
        trial = env.new_trial()

        stimulus = CATCH_SESSION_KEY if trial["catch"] else trial["stim_types"]

        # Unpack tuples with a single element.
        occurrences.setdefault(stimulus, 0.0)
        occurrences[stimulus] += 1.0

        # Store sequential occurrences, taking multi-sensory tasks into account.
        for modality, count in trial["sequential_count"].items():
            sequential_counts.setdefault(modality, [])
            sequential_counts[modality].append(count)

    occurrences = {k: v / trials for k, v in occurrences.items()}

    return occurrences, sequential_counts


@pytest.fixture
def default_env() -> AnnubesEnv:
    """Fixture for creating a default AnnubesEnv instance."""
    session = {"a": 0.5, "v": 0.5}
    catch_prob = 0.5
    return AnnubesEnv(session, catch_prob)  # type: ignore[arg-type]


@pytest.fixture
def custom_env() -> AnnubesEnv:
    """Fixture for creating a custom AnnubesEnv instance with specific parameters."""
    return AnnubesEnv(
        session={"v": 1},
        catch_prob=0.3,
        intensities={"v": [0.5, 1.0]},
        stim_time=800,
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
):
    """Test various types of fix_time specifications."""
    session = {"a": 0.5, "v": 0.5}
    catch_prob = 0.5

    env = AnnubesEnv(session, catch_prob, fix_time=time, iti=time)  # type: ignore[arg-type]
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


@pytest.mark.parametrize(
    ("catch_prob", "error_type"),
    [
        (0.0, None),
        (0.3, None),
        (0.7, None),
        (1.0, ValueError),
    ],
)
def test_catch_prob(catch_prob: float, error_type: type[Exception] | None):
    """Test if the catch trial probability is working as expected.

    This test checks:
    1. If the observed probability of catch trials matches the specified probability
    2. If the probability works correctly for various values including edge cases (0.0 and 1.0)
    """
    n_trials = 1000
    session = {"a": 0.5, "v": 0.5}

    if error_type is not None:
        with pytest.raises(Exception) as e:
            env = AnnubesEnv(
                session=session,  # type: ignore[arg-type]
                catch_prob=catch_prob,
                random_seed=RND_SEED,
            )
        assert e.type is error_type, f"incorrect error type raised: expected {error_type}, got {e.type}."
        return

    env = AnnubesEnv(session, catch_prob, random_seed=RND_SEED)  # type: ignore[arg-type]
    catch_count = 0

    for _ in range(n_trials):
        env.reset()
        trial_info = env.trial
        if trial_info is not None and trial_info.get("catch"):
            catch_count += 1

    observed_prob = catch_count / n_trials
    expected_prob = catch_prob

    # Check if the observed probability is close to the expected probability.
    assert np.isclose(
        observed_prob,
        expected_prob,
        rtol=0.05,
        atol=0.05,
    ), f"Expected catch probability {expected_prob}, but got {observed_prob}"


# TODO: Deal with cases where the max_sequential value is too low.
# This can affect the probability distribution.
# One option might be to start with a higher tolerance (final assertion)
# and gradually tighten it as `max_sequential` increases.
@pytest.mark.parametrize(
    ("session", "catch_prob", "max_sequential", "error_type"),
    [
        ({"v": 0.5, "a": 0.5}, 0.0, None, None),
        ({"v": 0.5, "a": 0.5}, 0.5, None, None),
        ({"v": 0.5, "a": 0.5}, 0.0, 10, None),
        ({"v": 0.5, "a": 0.5}, 0.5, 10, None),
        ({"v": 0.5, ("a", "v"): 0.5}, 0.0, None, None),
        ({"v": 0.5, ("a", "v"): 0.5}, 0.5, None, None),
        ({("v", "a", "o"): 1.0}, 0.5, None, None),
        ({"v": 0.0, "a": 0.5, "o": 0.9}, 0.5, None, None),
        # Tests that should fail.
        # ==================================================
        # All probabilities are 0.
        ({"v": 0.0, "a": 0.0, "o": 0.0}, 0.5, None, ValueError),
        # The session dictionary contains the 'catch' reserved keyword.
        ({"v": 0.5, "a": 0.5, CATCH_KEYWORD: 0.5}, 0.5, None, ValueError),
        # The session dictionary contains the 'CATCH' (upper case) reserved keyword.
        ({"v": 0.5, "a": 0.5, CATCH_KEYWORD.upper(): 0.5}, 0.5, None, ValueError),
        # The session is not a dictionary.
        (set(), 0.5, None, TypeError),
        # Max. sequential imposed on modality which should be presented in every trial.
        ({"v": 1.0, ("v", "a"): 0.5}, 0.0, 5, ValueError),
        # The catch probability is 1 but there are stimuli with non-zero probabilities.
        ({"v": 0.5, "a": 0.5}, 1.0, None, ValueError),
        # The catch probability is 1 but there is a limit on catch trials.
        ({"v": 0.0, "a": 0.0}, 1.0, {CATCH_KEYWORD: 5}, ValueError),
    ],
)
def test_annubes_env_probabilities_and_counts(
    session: dict,
    catch_prob: float,
    max_sequential: dict[str, int | None] | int | None,
    error_type: type[Exception] | None,
):
    """Test AnnubesEnv with various values for the session, catch probability and max_sequential.

    The test performs the following checks:
    1. Ensures that no sequence of non-catch trials longer than `max_sequential`
       of the same type occurs.
    2. Verifies that all specified trial types in the session occur.
    3. Checks if the distribution of trial types is roughly balanced
       according to the specified probabilities.
    4. Verifies that the frequency of catch trials matches the specified
       `catch_prob`.
    """
    # Ensure that the environment fails to instantiate
    # and the correct error type is raised.
    if error_type is not None:
        with pytest.raises(Exception) as e:
            env = AnnubesEnv(
                session=session,
                catch_prob=catch_prob,
                max_sequential=max_sequential,
                random_seed=RND_SEED,
            )
        assert e.type is error_type, f"incorrect error type raised: expected {error_type}, got {e.type}."
        return

    # We are past the failure checks, create
    # an environment with the specified parameters.
    env = AnnubesEnv(
        session=session,
        catch_prob=catch_prob,
        max_sequential=max_sequential,
        random_seed=RND_SEED,
    )

    # Collect trial statistics and consecutive presentation counts.
    # A dictionary of all occurrences of all combinations
    # that are expected to show up in the trials.
    occurrences, sequential_counts = _collect_trial_stats(env, N_TRIALS)

    # Assert that all stimuli are presented.
    # Some keys in the session might have a probability
    # of 0, so we have to exclude those.
    session_modalities: set = {k for k, v in env.session.items() if v > 0.0}
    if catch_prob > 0.0:
        session_modalities.add(CATCH_SESSION_KEY)
    occurrence_modalities = set(occurrences.keys())
    assert session_modalities == occurrence_modalities, "The trials do not reflect the session."

    # Assert that the stimuli occur with the right probabilities.
    for stimulus, probability in env.session.items():
        if probability > 0.0:
            # Multiply by the complementary of the catch trial probability
            # to obtain the right probability for this stimulus.
            comp_prob = probability * (1 - catch_prob)

            # TODO: Find a better way to compare these, 10% is a lot.
            assert stimulus in occurrences
            assert np.isclose(
                comp_prob,
                occurrences[stimulus],
                atol=0.1 * comp_prob,
                rtol=0.1 * comp_prob,
            )

        else:
            assert stimulus not in occurrences

    # Check for sequences longer than max_sequential.
    if any(v is not None for v in env.max_sequential.values()):
        sequential_counts = {k: np.array(v[1:]) for k, v in sequential_counts.items()}
        for modality, sequences in sequential_counts.items():
            if env.max_sequential[modality] is None:
                # Skip if there is no limit for this modality.
                continue
            # Get the indices of trials where the stimulus occurred more than max_sequential times in a row.
            error_idx = np.argwhere(np.array(sequences, dtype=np.uint32) > env.max_sequential[modality])  # type: ignore[operator]
            assert len(error_idx) == 0, (
                f"Found a sequence longer than {env.max_sequential[modality]} at trials {error_idx}"
            )


def test_observation_space(default_env: AnnubesEnv, custom_env: AnnubesEnv) -> None:
    """Test the observation space of both default and custom environments.

    This test checks:
    1. The shape of the observation space
    2. The names assigned to each dimension of the observation space
    """
    assert default_env.observation_space.shape == (4,)
    assert custom_env.observation_space.shape == (3,)

    assert default_env.observation_space.name == {  # type: ignore[attr-defined]
        "fixation": 0,
        "start": 1,
        **{trial: i for i, trial in enumerate(sorted(default_env.modalities), 2)},
    }
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
        session = {"a": 0.5, "v": 0.5}
        catch_prob = 0.5
        env1 = AnnubesEnv(session, catch_prob, random_seed=RND_SEED)  # type: ignore[arg-type]
        env2 = AnnubesEnv(session, catch_prob, random_seed=RND_SEED)  # type: ignore[arg-type]
        trial1 = env1._new_trial()
        trial2 = env2._new_trial()
        assert trial1 == trial2
