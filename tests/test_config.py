"""Test the Pydantic-based configuration."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import tomlkit as tkit

from neurogym.config import Config

rng = np.random.default_rng()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files.

    Returns:
        A path to a temporary directory that is removed after the test.
    """
    with tempfile.TemporaryDirectory() as _temp_dir:
        yield Path(_temp_dir)


def test_config_custom_toml_file(temp_dir: Path):
    """Test if a configuration can be created with a custom TOML file."""
    # Path to the configuration file
    config_file = temp_dir / "config.toml"

    try:
        # Set the level to some value that we can test
        level = "NONSENSETESTLEVEL"

        # Create a TOML file with an entry for the log level
        # ==================================================
        doc = tkit.document()
        doc.add(tkit.comment("This is a Neurogym configuration file."))
        doc.add(tkit.nl())

        monitor = tkit.table()
        log = tkit.table()
        monitor.append("log", log)
        log.add("level", level)
        doc.add("monitor", monitor)

        # Store the file
        # ==================================================
        with Path.open(config_file, "w") as fp:
            tkit.dump(doc, fp)

        # Create a new configuration
        # ==================================================
        config = Config(config_file)

        # Check that the log level in the configuration
        # is the same as the `level` variable
        assert config.monitor.log.level == level

    finally:
        # Clean up
        config_file.unlink(missing_ok=True)


@pytest.mark.parametrize("trigger", ["trial", "step"])
def test_config_instantiate_from_dict(
    temp_dir: Path,
    trigger: str,
):
    """Test if a custom configuration can be created from a nested dictionary."""
    # Random options
    local_dir = temp_dir / "local-test"
    save_interval = rng.integers(50, 100)
    plot_steps = rng.integers(5, 10)
    log_level = "WARNING"

    opts = {
        "local_dir": local_dir,
        "monitor": {
            "trigger": trigger,
            "interval": save_interval,
            "plot": {
                "value": plot_steps,
            },
            "log": {
                "level": log_level,
            },
        },
    }

    # Create a new configuration
    # ==================================================
    config = Config(**opts)  # type: ignore[arg-type]

    # Check that the configuration matches the variables above.
    # NOTE: Add tests for all nested options.
    assert config.local_dir == local_dir
    assert config.monitor.interval == save_interval
    assert config.monitor.trigger == trigger
    assert config.monitor.log.level == log_level


@pytest.mark.parametrize(
    ("input_dict", "expected_title"),
    [
        # Case 1: title is explicitly provided
        ({"env": {"name": "TrialEnv"}, "monitor": {"plot": {"title": "CustomTitle"}}}, "CustomTitle"),
        # Case 2: title is omitted — should fall back to env.name
        ({"env": {"name": "TrialEnv"}, "monitor": {"plot": {}}}, "TrialEnv"),
    ],
)
def test_monitor_plot_title_resolution_from_dict(input_dict, expected_title):
    """Test that monitor.plot.title resolves correctly from dict input."""
    config = Config(**input_dict)
    assert config.monitor.plot.title == expected_title
