# ruff: noqa: I001 - Import block is un-sorted or un-formatted
# fixing rule above leads to problems in pytest
# FIXME: figure out why this is the case and solve

from neurogym.core import BaseEnv, TrialEnv
from neurogym.utils import spaces
from neurogym.envs.registration import all_envs

__version__ = "2.0.0"
