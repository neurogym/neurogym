from collections.abc import Callable
from typing import Container

import numpy as np
import pytest

from neurogym.envs.native.annubes import AnnubesEnv

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
        rtol=0.05,
    ), f"Expected catch probability {expected_prob}, but got {observed_prob}"


# TODO: Deal with cases where the max_sequential value is too low.
# This can affect the probability distribution.
# One option might be to start with a higher tolerance (final assertion)
# and gradually tighten it as `max_sequential` increases.
@pytest.mark.parametrize("max_sequential", [None, 5, 10, 15])
@pytest.mark.parametrize("catch_prob", np.linspace(0.0, 0.5, 5))
@pytest.mark.parametrize(
    "session",
    [
        # Modalities with equal probabilities summing up to 1
        {"v": 0.5, "a": 0.5},
        # Modalities with non-equal probabilities summing up to 1
        {"v": 0.3, "a": 0.7},
        # Multiple modalities with overlap.
        {"v": 0.3, ("a", "v"): 0.7},
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
        # Session containing a combination of multiple
        # modalities to be presented in all trials.
        {("v", "a", "o"): 1.0},
    ],
)
def test_annubes_env_probabilities_and_counts(
    max_sequential: int,
    catch_prob: float,
    session: dict,
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

    # Helper functions
    # ==================================================
    def _is_close(
        x: float,
        tgt: float,
        atol=10 * np.finfo(float).eps,
        rtol=10 * np.finfo(float).eps,
    ) -> bool:
        """
        Check if two numbers are numerically close.

        Args:
            x: The number being checked.
            tgt: The target number.
            atol: Absolute tolerance. Defaults to 10*np.finfo(float).eps.
            rtol: Relative tolerance. Defaults to 10*np.finfo(float).eps.

        Returns:
            An indication of whether the two numbers are close up to
            the specified tolerance.
        """
        return np.isclose(x, tgt, atol=atol, rtol=rtol)

    # Unpack tuples recursively and unify into a single tuple without repetitions.
    def _tuple_or_string(x: None | str | Container) -> tuple:
        if x is None or isinstance(x, str):
            # Not a container
            return x
        elif len(x) == 1:
            # Some container with a single element
            return x[0]
        # A container with more than one element
        combined = set()
        for elem in x:
            combined.update(set(elem))
        return tuple(sorted(combined))

    # Collect results from multiple trials.
    def _collect_rollouts(runs: int, env: AnnubesEnv) -> dict:
        # Occurrences and sequential counts
        rollouts = dict.fromkeys(tuple(env.session.keys()) + (None,), 0.0)
        counts = {k: [] for k in tuple(env.stim_types) + (None,)}

        for _ in range(runs):
            trial = env.new_trial()

            stim_types = trial["stim_types"]
            if trial["catch"]:
                stim_types = None
            else:
                stim_types = _tuple_or_string(stim_types)

            # Unpack tuples with a single element.
            rollouts[stim_types] += 1.0

            # Store sequential occurrences, taking multi-sensory tasks into account
            for stim_type, seq_count in trial["sequential_count"].items():
                counts[stim_type].append(seq_count)

        rollouts = {k: v / runs for k, v in rollouts.items()}

        return rollouts, counts

    def _max_sequential_conflict(
        stim_types: tuple,
        catch_prob: float,
        session: dict,
        max_sequential: dict | int | None,
    ) -> bool:
        """
        Check the satisfiability of the the max_sequential condition.

        NOTE: This is a slightly modified version of the
        AnnubesEnv._prepare_max_sequential() method.

        Args:
            stim_types: A list of elementary stimulus types identified from the session.
            catch_prob: The catch probability.
            session: A session.
            max_sequential: An optional dictionary or integer specifying the
                number of consecutive stimulus presentations.

        Returns:
            An indication of whether the max_sequential condition can be satisfied.
        """

        default_max_sequential = max_sequential if isinstance(max_sequential, int) else None
        if not isinstance(max_sequential, dict):
            max_sequential = {}

        # Now populate max_sequential, including for catch trials ('None' key)
        for _stim_type in stim_types:
            max_sequential.setdefault(_stim_type, default_max_sequential)
        max_sequential.setdefault(None, default_max_sequential)

        # Check if the max_sequential conditions can be satisfied at all.
        # This could happen if the catch probability is 0 and one or more
        # modalities are supposed to appear 100% of the time but
        # at the same time max_sequential imposes a limit on how many times
        # that stimulus can be presented.
        if not _is_close(catch_prob, 0.0):
            return False

        stim_probs = {k: 0.0 for k in stim_types}
        for _stim_types, prob in session.items():
            if not isinstance(_stim_types, tuple):
                _stim_types = (_stim_types,)
            for _stim_type in _stim_types:
                stim_probs[_stim_type] += prob

        return any([max_sequential[k] is not None and _is_close(v, 1.0) for k, v in stim_probs.items()])

    # / Helper functions
    # ==================================================

    # Sum of the session probabilities.
    total = sum(session.values())

    # Ensure that the environment fails to instantiate
    # if all probability weights are 0.
    # ==================================================
    if _is_close(total, 0.0):
        with pytest.raises(ValueError) as e:
            env = AnnubesEnv(
                session=session,
                catch_prob=catch_prob,
                max_sequential=max_sequential,
                random_seed=RND_SEED,
            )
        assert str(e.value) == "Please ensure that at least one modality has a non-zero probability."
        return

    # Session with normalised probabilities
    norm_session = {_tuple_or_string(k): v / total for k, v in session.items()}

    # Extract the elementary modalities.
    stim_types = _tuple_or_string(list(session.keys()))

    # Check for max_sequential satisfiability and fail predictably
    # if it does not pass.
    # ==================================================
    if _max_sequential_conflict(stim_types, catch_prob, norm_session, max_sequential):
        with pytest.raises(ValueError) as e:
            env = AnnubesEnv(
                session=session,
                catch_prob=catch_prob,
                max_sequential=max_sequential,
                random_seed=RND_SEED,
            )
        assert (
            str(e.value)
            == "Invalid settings: max_sequential imposes a limit on a stimulus that should appear in every trial."
        )
        return

    # We are past the failure checks, create
    # an environment with the specified parameters.
    # ==================================================
    env = AnnubesEnv(
        session=session,
        catch_prob=catch_prob,
        max_sequential=max_sequential,
        random_seed=RND_SEED,
    )

    # Collect rollouts and consecutive counts
    # ==================================================
    # A dictionary of all occurrences of all combinations
    # that are expected to show up in the rollouts.
    rollouts, sequential_counts = _collect_rollouts(N_TRIALS, env)

    # Assert that all modality combinations occur.
    # Some keys in the session might have a probability
    # of 0, so we have to exclude those.
    # ==================================================
    assert len(set([k for k, v in session.items() if v > 0.0]).symmetric_difference(set(rollouts.keys()))), (
        "The trials do not reflect the session."
    )

    # Assert that the modalities occur with
    # the right probabilities.
    # ==================================================
    for combination, probability in norm_session.items():
        # Standardise the key
        combination = _tuple_or_string(combination)

        # Multiply by the complementary of the catch trial probability
        # to obtain the right probability for this combination
        comp_prob = probability * (1 - catch_prob)

        assert _is_close(
            comp_prob,
            rollouts[combination],
            atol=0.185 * comp_prob,
            rtol=0.185 * comp_prob,
        )

    # Check for sequences longer than max_sequential,
    # excluding catch trials.
    # ==================================================
    if any([v is not None for v in env.max_sequential.values()]):
        sequential_counts = {k: np.array(v[1:]) for k, v in sequential_counts.items()}
        for stim_type, sequences in sequential_counts.items():
            # Get the indices of trials where the stimulus
            # occurred more than max_sequential times in a row.
            error_idx = np.argwhere(np.array(sequences, dtype=np.uint32) > env.max_sequential[stim_type])
            assert len(error_idx) == 0, (
                f"Found a sequence longer than {env.max_sequential[stim_type]} at trials {error_idx}"
            )


def test_observation_space(default_env: AnnubesEnv, custom_env: AnnubesEnv) -> None:
    """Test the observation space of both default and custom environments.

    This test checks:
    1. The shape of the observation space
    2. The names assigned to each dimension of the observation space
    """
    assert default_env.observation_space.shape == (4,)
    assert custom_env.observation_space.shape == (3,)

    assert default_env.observation_space.name == {
        "fixation": 0,
        "start": 1,
        **{trial: i for i, trial in enumerate(sorted(default_env.stim_types), 2)},
    }  # type: ignore[attr-defined]
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
