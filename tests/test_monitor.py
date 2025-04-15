import tempfile
from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces

from neurogym.core import TrialEnv
from neurogym.wrappers.monitor import Monitor


class DummyEnv(TrialEnv):
    """Simple environment for testing the Monitor wrapper."""

    def __init__(self, dt: int = 100):
        super().__init__(dt=dt)
        self.observation_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.rewards = {0: -0.1, 1: 1.0}

    def _new_trial(self, **kwargs):
        """Start a new trial."""
        return {}

    def _step(self, action: int):
        """Take a step in the environment."""
        obs = self.observation_space.sample()
        reward = self.rewards[action]
        info = {"new_trial": True, "gt": action}
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, info


@pytest.fixture
def temp_folder():
    """Create a temporary directory for test files.

    Returns:
        A path to a temporary directory that is removed after the test.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


def test_monitor_initialization(temp_folder: str):
    """Test that Monitor initializes correctly with various parameters.

    Args:
        temp_folder: Temporary directory for saving monitor data.
    """
    env = DummyEnv()
    monitor = Monitor(env, folder=temp_folder, sv_per=10, sv_stp="trial")

    # Check that monitor attributes are set correctly
    assert monitor.sv_per == 10
    assert monitor.sv_stp == "trial"
    assert monitor.folder == temp_folder
    assert monitor.data == {"action": [], "reward": [], "performance": []}
    assert monitor.num_tr == 0


def test_monitor_data_collection():
    """Test that Monitor collects data correctly during environment steps."""
    env = DummyEnv()
    monitor = Monitor(env, sv_per=100, sv_stp="trial", verbose=False)

    # Reset the environment
    monitor.reset()

    # Take a few steps
    for _ in range(5):
        action = 1  # Always choose the rewarding action
        _, _, _, _, info = monitor.step(action)

        # Check if a new trial occurred
        if info.get("new_trial", False):
            assert len(monitor.data["action"]) > 0
            assert len(monitor.data["reward"]) > 0
            assert 1 in monitor.data["action"]  # Our chosen action
            assert 1.0 in monitor.data["reward"]  # Reward for action 1


@pytest.mark.parametrize("sv_stp", ["trial", "timestep"])
def test_monitor_save_data(temp_folder: str, sv_stp: str):
    """Test that Monitor saves data correctly to disk.

    Args:
        temp_folder: Temporary directory for saving monitor data.
        sv_stp: Save step type, either "trial" or "timestep".
    """
    env = DummyEnv()
    monitor = Monitor(
        env,
        folder=temp_folder,
        sv_per=3,  # Save after 3 trials/timesteps
        sv_stp=sv_stp,
        verbose=False,
    )

    # Reset and take steps until data is saved
    monitor.reset()

    # Take enough steps to trigger a save
    for _ in range(10):
        action = monitor.action_space.sample()
        monitor.step(action)

    # Check if files were created
    saved_files = list(Path(temp_folder).glob("*.npz"))
    assert len(saved_files) > 0, "No data files were saved"

    # Load and check a saved file
    data = np.load(saved_files[0])
    assert "action" in data
    assert "reward" in data


def test_evaluate_policy():
    """Test that evaluate_policy correctly evaluates policy performance."""
    env = DummyEnv()
    monitor = Monitor(env, verbose=False)

    # Create a simple mock model that always takes action 1
    class MockModel:
        def predict(self, observation, state=None, episode_start=None, deterministic=True):  # noqa: ARG002
            return 1, None  # Always return action 1 and no state

    mock_model = MockModel()

    # Test with mock model (always chooses rewarding action)
    num_trials = 10
    results_model = monitor.evaluate_policy(num_trials=num_trials, model=mock_model, verbose=False)

    # Check results structure
    assert "mean_performance" in results_model
    assert "mean_reward" in results_model
    assert "performances" in results_model
    assert "rewards" in results_model

    # With our mock model always choosing action 1 (which gives reward 1.0),
    # mean_reward should be close to 1.0
    assert results_model["mean_reward"] == 1.0
    assert len(results_model["rewards"]) == num_trials

    # Test with random policy (no model provided)
    results_random = monitor.evaluate_policy(num_trials=num_trials, model=None, verbose=False)

    # Check results structure
    assert "mean_performance" in results_random
    assert "mean_reward" in results_random
    assert "performances" in results_random
    assert "rewards" in results_random

    # With random policy in a binary action space, mean_reward should be around 0.5
    # but could vary, so we just check it's a valid value between 0 and 1
    assert 0 <= results_random["mean_reward"] <= 1.0
    assert len(results_random["rewards"]) == num_trials


def test_plot_training_history(temp_folder: str):
    """Test that plot_training_history generates a visualization.

    Args:
        temp_folder: Temporary directory for saving monitor data.
    """
    # Create environment and collect some data
    env = DummyEnv()
    monitor = Monitor(env, folder=temp_folder, sv_per=5, verbose=False)

    monitor.reset()
    for _ in range(20):
        action = 1  # Always choose rewarding action for predictable results
        monitor.step(action)

    # Generate plot
    fig = monitor.plot_training_history(save_fig=True)

    # Check if plot was created
    assert fig is not None, "Plot was not created"

    # Check if plot file exists
    plot_file = Path(temp_folder) / f"{env.unwrapped.__class__.__name__}_training_history.png"
    assert plot_file.exists(), "Plot file was not saved"
