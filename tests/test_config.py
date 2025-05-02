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


@pytest.mark.parametrize("plot_trigger", ["trials", "steps"])
def test_config_instantiate_from_dict(
    temp_dir: Path,
    plot_trigger: str,
):
    """Test if a custom configuration can be created from a nested dictionary."""
    # Random options
    _local_dir = temp_dir / "local-test"
    _plot_interval = rng.integers(0, 100)
    _log_level = "WARNING"

    opts = {
        "local_dir": _local_dir,
        "monitor": {
            "plot": {
                "interval": _plot_interval,
                "trigger": plot_trigger,
            },
            "log": {
                "level": _log_level,
            },
        },
    }

    # Create a new configuration
    # ==================================================
    config = Config(**opts)  # type: ignore[arg-type]

    # Check that the configuration matches the variables above.
    # NOTE: Add tests for all nested options.
    assert config.local_dir == _local_dir
    assert config.monitor.plot.interval == _plot_interval
    assert config.monitor.plot.trigger == plot_trigger
    assert config.monitor.log.level == _log_level


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
