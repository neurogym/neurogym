# --------------------------------------
from pathlib import Path

# --------------------------------------
from neurogym.config.base import ConfBase

# The root of the repository
ROOT_DIR = Path(__file__).parent.parent.parent.parent
PACKAGE_DIR = ROOT_DIR / "neurogym"
LOCAL_DIR = ROOT_DIR / "local"
CONFIG_DIR = PACKAGE_DIR / "config"

class PathConf(ConfBase):
    """
    Core path configuration.
    """

    root_dir: Path = ROOT_DIR
    package_dir: Path = PACKAGE_DIR
    config_dir: Path = CONFIG_DIR
    local_dir: Path = LOCAL_DIR
