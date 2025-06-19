# ruff: noqa: I001 - Import block is un-sorted or un-formatted
# fixing rule above leads to problems in pytest
# FIXME: figure out why this is the case and solve

from neurogym.utils import spaces
from neurogym.envs.registration import make
from neurogym.envs.registration import register
from neurogym.envs.registration import all_envs
from neurogym.envs.registration import all_tags
from neurogym.envs.collections import get_collection
from neurogym.utils.data import Dataset

__version__ = "2.1.0"
