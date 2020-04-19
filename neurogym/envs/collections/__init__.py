"""Collections of envs.

Each collection is a list of envs.
"""
from inspect import getmembers, isfunction
import importlib


def _collection_from_file(fname):
    """Return list of envs from file."""
    lib = 'neurogym.envs.collections.' + fname
    module = importlib.import_module(lib)
    envs = [name for name, val in getmembers(module) if isfunction(val)]
    envs = [env + '-v0' for env in envs if env[0] != '_']
    return envs


def get_collection(collection):
    if collection == '':
        return []  # placeholder for named collections
    else:
        try:
            return _collection_from_file(collection)
        except ModuleNotFoundError:
            raise ValueError('Unknown collection of envs, {}'.format(collection))