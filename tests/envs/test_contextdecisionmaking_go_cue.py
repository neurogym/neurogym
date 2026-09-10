"""Regression tests for the observable decision cue and preserved task behavior.

Controlled rollouts start a fresh trial with new_trial and its first observation,
then use the public step API to check action, observation and reward alignment.
"""

import numpy as np
import pytest

from neurogym.envs.native.contextdecisionmaking import ContextDecisionMaking
from neurogym.utils import TruncExp

MODES = [(False, 0), (False, 1), (True, 0), (True, 1)]
TRIAL = {"ground_truth": 1, "other_choice": 2, "coh_1": 50, "coh_2": 50}


def make_env(explicit=False, modality=0, **kwargs):
    """Construct a fresh deterministic task with multiple steps in each period."""
    options = {
        "dt": 100,
        "use_expl_context": explicit,
        "impl_context_modality": modality,
        "dim_ring": 2,
        "sigma": 0,
        "timing": {"fixation": 200, "stimulus": 200, "delay": 200, "decision": 200},
    }
    options.update(kwargs)
    env = ContextDecisionMaking(**options)
    # TrialEnv.seed also seeds callable timing distributions; reset(seed) does not.
    env.seed(17)
    return env


@pytest.mark.parametrize(("explicit", "modality"), MODES)
@pytest.mark.parametrize("dim_ring", [2, 4])
def test_fixation_marks_all_predecision_samples(explicit, modality, dim_ring):
    env = make_env(explicit, modality, dim_ring=dim_ring)
    try:
        env.new_trial(**TRIAL, context=modality)
        # These counts follow the literal 200 ms periods and dt=100 fixture.
        np.testing.assert_array_equal(env.ob[:, 0], [1, 1, 1, 1, 1, 1, 0, 0])
        np.testing.assert_array_equal(env.gt, [0, 0, 0, 0, 0, 0, 1, 1])
        assert env.ob.shape == (8, 1 + 2 * dim_ring + 2 * int(explicit))
        assert env.action_space.n == 1 + dim_ring
        for period in ("fixation", "stimulus", "delay"):
            np.testing.assert_array_equal(env.view_ob(period)[:, 0], [1, 1])
        np.testing.assert_array_equal(env.view_ob("decision")[:, 0], [0, 0])
    finally:
        env.close()


@pytest.mark.parametrize(
    ("fixation_ms", "delay_ms", "expected_boundary", "expected_delay_samples"),
    [(45, 0, 5, 0), (45, 19, 5, 0), (45, 59, 7, 2), (0, 0, 3, 0)],
)
def test_zero_length_periods_and_quantized_boundary(fixation_ms, delay_ms, expected_boundary, expected_delay_samples):
    # At dt=20, fixation45 becomes40, stimulus75 becomes60, delay59 becomes40.
    # Expected boundaries are declared by hand, not read from the implementation.
    env = make_env(
        dt=20,
        timing={"fixation": fixation_ms, "stimulus": 75, "delay": delay_ms, "decision": 60},
    )
    try:
        env.new_trial(**TRIAL, context=0)
        assert env.start_ind["decision"] == expected_boundary
        assert len(env.view_ob("delay")) == expected_delay_samples
        assert len(env.ob) == expected_boundary + 3
        np.testing.assert_array_equal(env.ob[:expected_boundary, 0], np.ones(expected_boundary))
        np.testing.assert_array_equal(env.ob[expected_boundary:, 0], [0, 0, 0])
        assert env.ob[expected_boundary - 1, 0] == 1
        assert env.ob[expected_boundary, 0] == 0
    finally:
        env.close()


@pytest.mark.parametrize(("explicit", "modality"), MODES)
def test_variable_delay_histories_diverge_exactly_at_shorter_go_cue(explicit, modality):
    env = make_env(
        explicit,
        modality,
        timing={"fixation": 100, "stimulus": 200, "delay": TruncExp(600, 100, 800), "decision": 200},
    )
    try:
        snapshots = {}
        for _ in range(64):
            # No stepping occurs between these trial generations: t remains zero.
            env.new_trial(**TRIAL, context=modality)
            boundary = int(env.start_ind["decision"])
            snapshots.setdefault(boundary, (env.ob.copy(), env.gt.copy()))
        assert len(snapshots) >= 2, "Fixture must draw distinct legal delay durations"
        boundary, longer_boundary = sorted(snapshots)[:2]
        short_ob, short_gt = snapshots[boundary]
        long_ob, long_gt = snapshots[longer_boundary]
        np.testing.assert_array_equal(short_ob[:boundary], long_ob[:boundary])
        assert (short_gt[boundary], long_gt[boundary]) == (1, 0)
        np.testing.assert_array_equal(short_ob[boundary, 1:], long_ob[boundary, 1:])
        assert short_ob[boundary, 0] == 0
        assert long_ob[boundary, 0] == 1
        assert not np.array_equal(short_ob[boundary], long_ob[boundary])
    finally:
        env.close()


@pytest.mark.parametrize(("explicit", "modality"), MODES)
def test_public_step_returns_go_cue_before_choice_is_scored(explicit, modality):
    env = make_env(explicit, modality)
    try:
        env.new_trial(**TRIAL, context=modality)
        observation = env.ob[0].copy()
        # The first six actions belong to fixation, stimulus and delay.
        for action_index in range(6):
            assert observation[0] == 1
            observation, reward, terminated, truncated, info = env.step(0)
            assert reward == 0
            assert not terminated
            assert not truncated
            assert not info["new_trial"]
            assert info["gt"] == 0  # Label for the action just taken, not returned ob.
            if action_index < 5:
                assert observation[0] == 1
        # Last delay action returns the first decision observation.
        assert env.t == 600
        assert observation[0] == 0
        _, reward, terminated, truncated, info = env.step(1)
        assert reward == 1
        assert not terminated
        assert not truncated
        assert info["new_trial"]
        assert info["gt"] == 1
        assert info["performance"] == 1
    finally:
        env.close()


@pytest.mark.parametrize(("explicit", "modality"), MODES)
def test_literal_sensory_context_and_label_controls(explicit, modality):
    env = make_env(explicit, modality)
    try:
        env.new_trial(**TRIAL, context=modality)
        # Opposing two-location stimuli at coherence50: independently specified.
        expected = [0.75, 0.25, 0.25, 0.75] if modality == 0 else [0.25, 0.75, 0.75, 0.25]
        np.testing.assert_allclose(env.ob[2:4, 1:5], [expected, expected], rtol=0, atol=1e-7)
        np.testing.assert_array_equal(env.ob[[0, 1, 4, 5, 6, 7], 1:5], np.zeros((6, 4)))
        np.testing.assert_array_equal(env.gt, [0, 0, 0, 0, 0, 0, 1, 1])
        if explicit:
            context = [1, 0] if modality == 0 else [0, 1]
            np.testing.assert_array_equal(env.ob[:, 5:7], np.tile(context, (8, 1)))
    finally:
        env.close()


@pytest.mark.parametrize("abort", [False, True])
@pytest.mark.parametrize("phase_start_index", [2, 4], ids=["stimulus", "delay"])
def test_cue_repair_preserves_ignored_stimulus_and_delay_actions(abort, phase_start_index):
    env = make_env(abort=abort)
    try:
        env.new_trial(**TRIAL, context=0)
        for _ in range(phase_start_index):
            _, _, _, _, info = env.step(0)
            assert not info["new_trial"]
        _, reward, terminated, truncated, info = env.step(2)
        assert reward == 0
        assert not terminated
        assert not truncated
        assert not info["new_trial"]
        assert info["gt"] == 0
    finally:
        env.close()


@pytest.mark.parametrize("abort", [False, True])
def test_initial_fixation_abort_contract_is_preserved(abort):
    env = make_env(abort=abort, rewards={"abort": -0.25, "correct": 2.5})
    try:
        env.new_trial(**TRIAL, context=0)
        _, reward, terminated, truncated, info = env.step(2)
        assert reward == -0.25
        assert not terminated
        assert not truncated
        assert info["new_trial"] == abort
        assert info["gt"] == 0
    finally:
        env.close()


@pytest.mark.parametrize(("action", "expected_reward", "expected_performance"), [(1, 2.5, 1), (2, 0, 0)])
def test_decision_reward_and_trial_termination_are_preserved(action, expected_reward, expected_performance):
    env = make_env(rewards={"abort": -0.25, "correct": 2.5})
    try:
        env.new_trial(**TRIAL, context=0)
        for _ in range(6):
            _, _, _, _, info = env.step(0)
            assert not info["new_trial"]
        _, reward, terminated, truncated, info = env.step(action)
        assert reward == expected_reward
        assert not terminated
        assert not truncated
        assert info["new_trial"]
        assert info["gt"] == 1
        assert info["performance"] == expected_performance
    finally:
        env.close()
