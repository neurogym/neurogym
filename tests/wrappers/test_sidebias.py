import re
from unittest.mock import Mock

import numpy as np
import pytest

import neurogym as ngym
from neurogym.wrappers.side_bias import SideBias


class MockTrialEnv(ngym.TrialEnv):
    """Mock environment for testing SideBias wrapper."""

    def __init__(self):
        super().__init__()
        self.choices = [0, 1]  # Binary choice task
        self.observation_space = Mock()
        self.action_space = Mock()

    def _new_trial(self, **kwargs):
        return {"ground_truth": kwargs.get("ground_truth", 0)}

    def _step(self, _action):
        return np.array([0, 0]), 0, False, False, {"new_trial": True}


def test_probs_validation():
    """Test that probs parameter validation works correctly."""
    env = MockTrialEnv()

    msg = (
        "probs must be a 2D list of lists (matrix) with shape (n_blocks, n_choices),"
        "e.g., probs = [[0.5, 0.5], [0.2, 0.8], [0.8, 0.2]] for n_blocks = 3 and n_choices = 2."
    )

    # Test 1: None probs should raise ValueError
    with pytest.raises(TypeError, match=re.escape(msg)):
        SideBias(env, probs=None)

    # Test 2: Float/int probs should raise TypeError (no auto-generation)
    with pytest.raises(TypeError, match=re.escape(msg)):
        SideBias(env, probs=0.8)

    # Test 3: Wrong dimensions should raise TypeError
    with pytest.raises(TypeError, match=re.escape(msg)):
        SideBias(env, probs=[0.8, 0.2])  # 1D array

    # Test 4: Wrong number of choices should raise ValueError
    with pytest.raises(ValueError):
        SideBias(env, probs=[[0.8, 0.1, 0.1], [0.2, 0.4, 0.4]])  # 3 choices, env has 2

    # Test 5: Rows not summing to 1 should raise ValueError
    with pytest.raises(ValueError):
        SideBias(env, probs=[[0.8, 0.1], [0.2, 0.9]])  # First row sums to 0.9, second to 1.1

    env.choices = [0, 1, 2]
    # Test 6: Negative probabilities should raise ValueError
    with pytest.raises(ValueError, match="All probabilities in probs must be non-negative."):
        SideBias(env, probs=[[0.8, 0.4, -0.2], [0.2, 0, 0.8]])

    env.choices = [0, 1]
    # Test 7: Valid probs should work
    wrapper = SideBias(env, probs=[[0.8, 0.2], [0.2, 0.8], [0.5, 0.5]])
    assert wrapper.choice_prob.shape == (3, 2)
    assert np.allclose(np.sum(wrapper.choice_prob, axis=1), 1.0)


def test_block_dur_validation():
    """Test that block_dur parameter validation works correctly."""
    env = MockTrialEnv()
    probs = [[0.8, 0.2], [0.2, 0.8]]

    # Test 1: Valid integer block_dur
    wrapper = SideBias(env, probs=probs, block_dur=200)
    assert wrapper.block_dur == 200

    # Test 2: Invalid integer (< 1) should raise ValueError
    with pytest.raises(ValueError):
        SideBias(env, probs=probs, block_dur=0)

    with pytest.raises(ValueError):
        SideBias(env, probs=probs, block_dur=-10)

    # Test 3: Valid float block_dur (probability)
    wrapper = SideBias(env, probs=probs, block_dur=0.1)
    assert wrapper.block_dur == 0.1

    # Test 4: Invalid float (not in (0,1)) should raise ValueError
    with pytest.raises(ValueError):
        SideBias(env, probs=probs, block_dur=0.0)

    with pytest.raises(ValueError):
        SideBias(env, probs=probs, block_dur=1.0)

    with pytest.raises(ValueError):
        SideBias(env, probs=probs, block_dur=1.5)

    # Test 5: Valid tuple block_dur (range)
    wrapper = SideBias(env, probs=probs, block_dur=(100, 300))
    assert wrapper.block_dur == (100, 300)

    # Test 6: Invalid tuple length should raise ValueError
    with pytest.raises(ValueError, match="exactly 2 elements"):
        SideBias(env, probs=probs, block_dur=(100,))

    with pytest.raises(ValueError, match="exactly 2 elements"):
        SideBias(env, probs=probs, block_dur=(100, 200, 300))

    # Test 7: Invalid tuple values should raise errors
    with pytest.raises(TypeError, match="must be integers"):
        SideBias(env, probs=probs, block_dur=(100.5, 200))

    with pytest.raises(ValueError, match="low value must be >= 1"):
        SideBias(env, probs=probs, block_dur=(0, 200))

    with pytest.raises(ValueError, match="high value must be >= low"):
        SideBias(env, probs=probs, block_dur=(300, 200))

    # Test 8: Invalid type should raise TypeError
    with pytest.raises(TypeError, match="block_dur must be one of"):
        SideBias(env, probs=probs, block_dur="invalid")
    with pytest.raises(TypeError, match="block_dur must be one of"):
        SideBias(env, probs=probs, block_dur=[100, 200])


def test_block_switching_behavior():
    """Test block switching behavior for different block_dur types."""
    env = MockTrialEnv()
    probs = [[0.8, 0.2], [0.2, 0.8], [0.4, 0.6]]
    block_dur = 5

    # Test 1: Fixed duration block switching
    wrapper = SideBias(env, probs=probs, block_dur=block_dur)

    initial_block = wrapper.curr_block
    trials_in_same_block = 0

    # Run several trials and check block changes
    for i in range(10):
        wrapper.new_trial()
        if i < block_dur - 1:
            # First block_dur - 1 trials should be in same block
            assert wrapper.curr_block == initial_block
            trials_in_same_block += 1
        elif i == block_dur - 1:
            # block_dur - 1 trial should switch to different block
            assert wrapper.curr_block != initial_block
            break

    # Test 2: Probability-based switching (with fixed seed for determinism)
    env.seed(42)
    wrapper = SideBias(env, probs=probs, block_dur=0.5)  # 50% switch probability

    initial_block = wrapper.curr_block
    switches = 0

    # Run many trials and count switches
    for _ in range(100):
        old_block = wrapper.curr_block
        wrapper.new_trial()
        if wrapper.curr_block != old_block:
            switches += 1

    # With 50% probability, we expect roughly 50 switches in 100 trials
    # Allow some variance due to randomness
    assert 30 <= switches <= 70, f"Expected ~50 switches, got {switches}"

    # Test 3: Range-based block duration
    env.seed(123)
    wrapper = SideBias(env, probs=probs, block_dur=(2, 4))

    # Test that durations are within range by checking when blocks change
    block_changes = []
    current_block = wrapper.curr_block
    trial_count = 0

    for _ in range(50):
        wrapper.new_trial()
        trial_count += 1
        if wrapper.curr_block != current_block:
            block_changes.append(trial_count)
            current_block = wrapper.curr_block
            trial_count = 0

    # Check that block durations were in [2, 4] range
    for duration in block_changes:
        assert 2 <= duration <= 4, f"Block duration {duration} not in [2, 4]"
