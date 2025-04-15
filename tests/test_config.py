"""Test the Pydantic-based configuration."""

from pathlib import Path

import tomlkit as tkit

from neurogym.config import Config


def test_config_custom_toml_file():
    """Test if a configuration can be created with a custom TOML file."""
    # Path to the current directory
    cur_dir = Path(__file__).parent

    # Path to the configuration file
    config_file = cur_dir / "config.toml"

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
