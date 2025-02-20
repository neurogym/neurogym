from pathlib import Path

import neurogym as ngym
from neurogym.config.base import ConfBase


class PathConf(ConfBase):
    """Core path configuration."""

    root_dir: Path = ngym.ROOT_DIR
    package_dir: Path = ngym.PACKAGE_DIR
    config_dir: Path = ngym.CONFIG_DIR
    local_dir: Path = ngym.LOCAL_DIR
