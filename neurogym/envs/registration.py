import importlib
from inspect import getmembers, isclass, isfunction
from pathlib import Path

import gymnasium as gym
from packaging import version

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
        env_prefix = ""
    elif env_prefix[-1] != ".":
        env_prefix = env_prefix + "."

    if allow_list is None:
        allow_list = []

    # Root path of neurogym.envs folder
    env_root = Path(__file__).resolve().parent
    lib_root = "neurogym.envs."
    if foldername is not None:
        env_root = env_root / foldername
        lib_root = lib_root + foldername + "."

    # Only take .py files
    files = [p for p in env_root.iterdir() if p.suffix == ".py"]
    # Exclude files starting with '_'
    files = [f for f in files if f.name[0] != "_"]
    filenames = [f.name[:-3] for f in files]  # remove .py suffix
    filenames = sorted(filenames)

    env_dict = {}
    for filename in filenames:
        lib = lib_root + filename
        module = importlib.import_module(lib)
        for name, _val in getmembers(module):
            if name in allow_list:
                env_dict[env_prefix + name + "-v0"] = lib + ":" + name

    return env_dict


NATIVE_ALLOW_LIST = [
    "AntiReach",
    "Bandit",
    "ContextDecisionMaking",
    "DawTwoStep",
    "DelayComparison",
    "DelayMatchCategory",
    "DelayMatchSample",
    "DelayMatchSampleDistractor1D",
    "DelayPairedAssociation",
    # 'Detection',  # TODO: Temporary removing until bug fixed # noqa: ERA001
    "DualDelayMatchSample",
    "EconomicDecisionMaking",
    "GoNogo",
    "HierarchicalReasoning",
    "IntervalDiscrimination",
    "MotorTiming",
    "MultiSensoryIntegration",
    "Null",
    "OneTwoThreeGo",
    "PerceptualDecisionMaking",
    "PerceptualDecisionMakingDelayResponse",
    "PostDecisionWager",
    "ProbabilisticReasoning",
    "PulseDecisionMaking",
    "Reaching1D",
    "Reaching1DWithSelfDistraction",
    "ReachingDelayResponse",
    "ReadySetGo",
    "SingleContextDecisionMaking",
    # 'SpatialSuppressMotion',   # noqa: ERA001
    # TODO: raises ModuleNotFound error since requires scipy, which is not in the requirements of neurogym.
    # FIXME: I have added scipy to requirements (for other reason), does this mean SpatialSuppressMotion is valid?
    # 'ToneDetection'  # TODO: Temporary removing until bug fixed # noqa: ERA001
]
ALL_NATIVE_ENVS = _get_envs(
    foldername=None,
    env_prefix=None,
    allow_list=NATIVE_ALLOW_LIST,
)

_psychopy_prefix = "neurogym.envs.psychopy."
ALL_PSYCHOPY_ENVS = {
    "psychopy.RandomDotMotion-v0": _psychopy_prefix + "perceptualdecisionmaking:RandomDotMotion",
    "psychopy.VisualSearch-v0": _psychopy_prefix + "visualsearch:VisualSearch",
    "psychopy.SpatialSuppressMotion-v0": _psychopy_prefix + "spatialsuppressmotion:SpatialSuppressMotion",
}

_contrib_name_prefix = "contrib."
_contrib_prefix = "neurogym.envs.contrib."
CONTRIB_ALLOW_LIST = [
    # 'AngleReproduction',
    # 'CVLearning',
    # 'ChangingEnvironment',
    # 'IBL',
    # 'MatchingPenny',
    # 'MemoryRecall',
    # 'Pneumostomeopening'
]
ALL_CONTRIB_ENVS = _get_envs(
    foldername="contrib",
    env_prefix="contrib",
    allow_list=CONTRIB_ALLOW_LIST,
)


# Automatically register all tasks in collections
def _get_collection_envs():
    """Register collection tasks in collections folder.

    Each environment is named collection_name.env_name-v0
    """
    derived_envs = {}
    # TODO: Temporary disabling priors task
    collection_libs = ["perceptualdecisionmaking", "yang19"]
    for collection_lib in collection_libs:
        lib = "neurogym.envs.collections." + collection_lib
        module = importlib.import_module(lib)
        envs = [name for name, val in getmembers(module) if isfunction(val) or isclass(val)]
        envs = [env for env in envs if env[0] != "_"]  # ignore private members
        # TODO: check is instance gym.env
        env_dict = {f"{collection_lib}.{env}-v0": f"{lib}:{env}" for env in envs}
        valid_envs = get_collection(collection_lib)
        derived_envs.update({key: env_dict[key] for key in valid_envs})
    return derived_envs


ALL_COLLECTIONS_ENVS = _get_collection_envs()

ALL_ENVS = {**ALL_NATIVE_ENVS, **ALL_PSYCHOPY_ENVS, **ALL_CONTRIB_ENVS}

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
    env_list = sorted(envs.keys())
    if tag is None:
        return env_list
    if not isinstance(tag, str):
        msg = f"{type(tag)=} must be a string."
        raise TypeError(msg)

    new_env_list = []
    for env in env_list:
        from_, class_ = envs[env].split(":")
        imported = getattr(__import__(from_, fromlist=[class_]), class_)
        env_tag = imported.metadata.get("tags", [])
        if tag in env_tag:
            new_env_list.append(env)
    return new_env_list


def all_tags():
    return [
        "confidence",
        "context dependent",
        "continuous action space",
        "delayed response",
        "go-no-go",
        "motor",
        "multidimensional action space",
        "n-alternative",
        "perceptual",
        "reaction time",
        "steps action space",
        "supervised",
        "timing",
        "two-alternative",
        "value-based",
        "working memory",
    ]


def _distance(s0, s1):
    # Copyright (c) 2018 luozhouyang
    if s0 is None:
        msg = "Argument s0 is NoneType."
        raise TypeError(msg)
    if s1 is None:
        msg = "Argument s1 is NoneType."
        raise TypeError(msg)
    if s0 == s1:
        return 0.0
    if len(s0) == 0:
        return len(s1)
    if len(s1) == 0:
        return len(s0)

    v0 = [0] * (len(s1) + 1)
    v1 = [0] * (len(s1) + 1)

    for i in range(len(v0)):
        v0[i] = i

    for i in range(len(s0)):
        v1[0] = i + 1
        for j in range(len(s1)):
            cost = 1
            if s0[i] == s1[j]:
                cost = 0
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0, v1 = v1, v0

    return v0[len(s1)]


def make(id_, **kwargs):
    try:
        # TODO: disable gym 0.24 env_checker for now (raises warnings, even errors when ob not in observation_space)
        # FIXME: is this still relevant for gymnasium?
        if version.parse(gym.__version__) >= version.parse("0.24.0"):
            return gym.make(id_, disable_env_checker=True, **kwargs)
        return gym.make(id_, **kwargs)

    except gym.error.UnregisteredEnv as e:  # FIXME: check if this is still relevant when using gymnasium
        # backward compatibility with old versions of gym
        if hasattr(gym.envs.registry, "all"):
            all_ids = [env.id for env in gym.envs.registry.all()]
        else:
            all_ids = [env.id for env in gym.envs.registry.values()]

        dists = [_distance(id_, env_id) for env_id in all_ids]
        # Python argsort
        sort_inds = sorted(range(len(dists)), key=dists.__getitem__)
        env_guesses = [all_ids[sort_inds[i]] for i in range(5)]
        err_msg = f"No registered env with id_: {id_}.\nDo you mean:\n"
        for env_guess in env_guesses:
            err_msg += "    " + env_guess + "\n"
        raise gym.error.UnregisteredEnv(err_msg) from e


# backward compatibility with old versions of gym
if hasattr(gym.envs.registry, "all"):
    _all_gym_envs = [env.id for env in gym.envs.registry.all()]
else:
    _all_gym_envs = [env.id for env in gym.envs.registry.values()]


def register(id_, **kwargs) -> None:
    if id_ not in _all_gym_envs:
        gym.envs.registration.register(id=id_, **kwargs)


for env_id, entry_point in ALL_EXTENDED_ENVS.items():
    register(id_=env_id, entry_point=entry_point)
