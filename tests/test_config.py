"""Test the Pydantic-based configuration."""

from pathlib import Path

import tomlkit as tkit

import neurogym as ngym
from neurogym.config.conf import Conf


def test_load_default_config():
    """Test if the default configuration can be loaded."""
    ngym.logger.info(f"Logging level: {ngym.conf.log.level}")
    ngym.logger.info("Neurogym configuration loaded successfully.")


def test_config_custom_toml_file():
    """Test if a configuration can be created with a custom TOML file."""
    # Path to the current directory
    cur_dir = Path(__file__).parent

    # Path to the configuration file
    config_file = cur_dir / "settings.toml"

    try:
        # Set the level to some value that we can test
        level = "NONSENSETESTLEVEL"

        # Create a TOML file with an entry for the log level
        # ==================================================
        doc = tkit.document()
        doc.add(tkit.comment("This is a Neurogym configuration file."))
        doc.add(tkit.nl())

        log = tkit.table()
        log.add("level", level)
        doc.add("log", log)

        # Store the file
        # ==================================================
        with Path.open(config_file, "w") as fp:
            tkit.dump(doc, fp)

        # Create a new configuration
        # ==================================================
        class CustomConf(Conf):
            @classmethod
            def settings_file(cls):
                return config_file

        conf = CustomConf()

        # Check that the log level in the configuration
        # is the same as the `level` variable
        assert conf.log.level == level

    finally:
        # Clean up
        config_file.unlink(missing_ok=True)
