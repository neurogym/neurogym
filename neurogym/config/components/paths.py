from pathlib import Path

from neurogym.config.base import CONFIG_DIR, LOCAL_DIR, PACKAGE_DIR, ROOT_DIR, ConfBase



class PathConf(ConfBase):
    """Core path configuration."""

    root_dir: Path = ROOT_DIR
    package_dir: Path = PACKAGE_DIR
    config_dir: Path = CONFIG_DIR
    local_dir: Path = LOCAL_DIR
