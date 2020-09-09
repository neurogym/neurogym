import importlib
from inspect import getmembers, isfunction, isclass
from pathlib import Path

import gym
from gym.envs.registration import register

from neurogym.envs.collections import get_collection


def _get_envs(foldername=None, env_prefix=None, allow_list=None):
    """A helper function to get all environments in a folder.

    Example usage:
        _get_envs(foldername=None, env_prefix=None)
        _get_envs(foldername='contrib', env_prefix='contrib')

    The results still need to be manually cleaned up, so this is just a helper

    Args:
        foldername: str or None. If str, in the form of contrib, etc.
        env_prefix: str or None, if not None, add this prefix to all env ids
        allow_list: list of allowed env name, for manual curation
    """

    if env_prefix is None:
        env_prefix = ''
    else:
        if env_prefix[-1] != '.':
            env_prefix = env_prefix + '.'

    if allow_list is None:
        allow_list = list()

    # Root path of neurogym.envs folder
    env_root = Path(__file__).resolve().parent
    lib_root = 'neurogym.envs.'
    if foldername is not None:
        env_root = env_root / foldername
        lib_root = lib_root + foldername + '.'

    # Only take .py files
    files = [p for p in env_root.iterdir() if p.suffix == '.py']
    # Exclude files starting with '_'
    files = [f for f in files if f.name[0] != '_']
    filenames = [f.name[:-3] for f in files]  # remove .py suffix
    filenames = sorted(filenames)

    env_dict = {}
    for filename in filenames:
        # lib = 'neurogym.envs.collections.' + l
        lib = lib_root + filename
        module = importlib.import_module(lib)
        for name, val in getmembers(module):
            if name in allow_list:
                env_dict[env_prefix + name + '-v0'] = lib + ':' + name

    return env_dict


NATIVE_ALLOW_LIST = [
    'AntiReach',
    'Bandit',
    'ContextDecisionMaking',
    'DawTwoStep',
    'DelayComparison',
    'DelayMatchCategory',
    'DelayMatchSample',
    'DelayMatchSampleDistractor1D',
    'DelayPairedAssociation',
    'Detection',
    'DualDelayMatchSample',
    'EconomicDecisionMaking',
    'GoNogo',
    'HierarchicalReasoning',
    'IntervalDiscrimination',
    'MotorTiming',
    'MultiSensoryIntegration',
    'NAltPerceptualDecisionMaking',
    'Nothing',
    'OneTwoThreeGo',
    'PerceptualDecisionMaking',
    'PerceptualDecisionMakingDelayResponse',
    'PostDecisionWager',
    'ProbabilisticReasoning',
    'PulseDecisionMaking',
    'Reaching1D',
    'Reaching1DWithSelfDistraction',
    'ReachingDelayResponse',
    'ReadySetGo',
    'SingleContextDecisionMaking'
]
ALL_NATIVE_ENVS = _get_envs(foldername=None, env_prefix=None,
                            allow_list=NATIVE_ALLOW_LIST)

_psychopy_prefix = 'neurogym.envs.psychopy.'
ALL_PSYCHOPY_ENVS = {
    'psychopy.RandomDotMotion-v0':
        _psychopy_prefix + 'perceptualdecisionmaking:RandomDotMotion',
    'psychopy.VisualSearch-v0':
        _psychopy_prefix + 'visualsearch:VisualSearch',
    'psychopy.SpatialSuppressMotion-v0':
        _psychopy_prefix + 'spatialsuppressmotion:SpatialSuppressMotion',
}

_contrib_name_prefix = 'contrib.'
_contrib_prefix = 'neurogym.envs.contrib.'
CONTRIB_ALLOW_LIST = [
    'AngleReproduction',
    'CVLearning',
    'ChangingEnvironment',
    # 'IBL',
    'MatchingPenny',
    'MemoryRecall',
    'Pneumostomeopening'
]
ALL_CONTRIB_ENVS = _get_envs(foldername='contrib', env_prefix='contrib',
                             allow_list=CONTRIB_ALLOW_LIST)


# Automatically register all tasks in collections
def _get_collection_envs():
    """Register collection tasks in collections folder.

    Each environment is named collection_name.env_name-v0
    """
    derived_envs = {}
    collection_libs = ['perceptualdecisionmaking', 'yang19', 'priors']
    for l in collection_libs:
        lib = 'neurogym.envs.collections.' + l
        module = importlib.import_module(lib)
        envs = [name for name, val in getmembers(module) if isfunction(val) or isclass(val)]
        envs = [env for env in envs if env[0] != '_']  # ignore private members
        # TODO: check is instance gym.env
        env_dict = {l+'.'+env+'-v0': lib + ':' + env for env in envs}
        valid_envs = get_collection(l)
        derived_envs.update({key: env_dict[key] for key in valid_envs})
    return derived_envs


ALL_COLLECTIONS_ENVS = _get_collection_envs()

ALL_ENVS = {
    **ALL_NATIVE_ENVS, **ALL_PSYCHOPY_ENVS, **ALL_CONTRIB_ENVS
}

ALL_EXTENDED_ENVS = {**ALL_ENVS, **ALL_COLLECTIONS_ENVS}


def all_envs(tag=None, psychopy=False, contrib=False, collections=False):
    """Return a list of all envs in neurogym."""
    envs = ALL_NATIVE_ENVS.copy()
    if psychopy:
        envs.update(ALL_PSYCHOPY_ENVS)
    if contrib:
        envs.update(ALL_CONTRIB_ENVS)
    if collections:
        envs.update(ALL_COLLECTIONS_ENVS)
    env_list = sorted(list(envs.keys()))
    if tag is None:
        return env_list
    else:
        if not isinstance(tag, str):
            raise ValueError('tag must be str, but got ', type(tag))

        new_env_list = list()
        for env in env_list:
            from_, class_ = envs[env].split(':')
            imported = getattr(__import__(from_, fromlist=[class_]), class_)
            env_tag = imported.metadata.get('tags', [])
            if tag in env_tag:
                new_env_list.append(env)
        return new_env_list


_all_gym_envs = [env.id for env in gym.envs.registry.all()]
for env_id, entry_point in ALL_EXTENDED_ENVS.items():
    if env_id not in _all_gym_envs:
        register(id=env_id, entry_point=entry_point)


def all_tags():
    return ['confidence', 'context dependent', 'continuous action space', 'delayed response', 'go-no-go',
            'motor', 'multidimensional action space', 'n-alternative', 'perceptual', 'reaction time',
            'steps action space', 'supervised', 'timing', 'two-alternative', 'value-based', 'working memory']


__all__ = ['multisensory']
