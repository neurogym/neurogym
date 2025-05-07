# ruff: noqa: I001 - Import block is un-sorted or un-formatted
# fixing rule above leads to problems in pytest
# FIXME: figure out why this is the case and solve

from neurogym.config import Config
from neurogym.config import config
from neurogym.config import logger
from neurogym import utils
from neurogym.core import BaseEnv
from neurogym.core import TrialEnv
from neurogym.core import TrialWrapper
from neurogym.utils import spaces
from neurogym.envs.registration import make
from neurogym.envs.registration import register
from neurogym.envs.registration import all_envs
from neurogym.envs.registration import all_tags
from neurogym.envs.collections import get_collection
from neurogym.wrappers import all_wrappers
from neurogym.utils.data import Dataset
from neurogym.utils import ngym_random

__version__ = "2.0.0"
