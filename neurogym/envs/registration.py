import importlib
from inspect import getmembers, isclass, isfunction
from pathlib import Path

import gymnasium as gym

from neurogym.envs.collections import get_collection


def _get_envs(
    foldername: str | None = None,
    exclude: str | list[str] | None = EXCLUDE_ENVS,
) -> dict[str, str]:
    """Discover and register all environments from a specified folder.

    This function scans Python files in the specified folder, imports them, and registers
    all classes that inherit from `gym.Env`.

    Args:
        foldername: A string specifying the subfolder within `neurogym.envs` to search for
            environment files. If `None`, the function searches in the root folder of
            `neurogym.envs`. For example:
                - foldername=None: Searches in `neurogym/envs/`.
                - foldername="contrib": Searches in `neurogym/envs/contrib/`.
        exclude: A list of environment names to exclude from registration.

    Returns:
        A dictionary mapping environment IDs to their entry points.
    """
    # Validate exclude list
    exclude = [exclude] if isinstance(exclude, str) else exclude or []
    if not isinstance(exclude, list):
        msg = f"{type(exclude)=} must be a string or a list of strings."
        raise TypeError(msg)

    # Root path of neurogym.envs folder
    env_root = Path(__file__).resolve().parent
    lib_root = "neurogym.envs."
    if foldername is not None:
        env_root /= foldername
        lib_root += f"{foldername}."
    py_files = [p for p in env_root.iterdir() if p.suffix == ".py" and p.name != "__init__.py"]

    env_dict: dict[str, str] = {}
    for file in py_files:
        module_name = lib_root + file.stem
        module = importlib.import_module(module_name)

        # Discover all classes defined in the module that inherit from gym.Env
        for name, obj in getmembers(module, isclass):
            if issubclass(obj, gym.Env) and name not in exclude and obj.__module__ == module_name:
                env_id = f"{foldername}.{name}-v0" if foldername else f"{name}-v0"
                entry_point = f"{module_name}:{name}"
                env_dict[env_id] = entry_point
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
CONTRIB_ALLOW_LIST: list = [
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
# FIXME: Collection envs are defined in a slightly different way. Refactor them to allow using the function above.
def _get_collection_envs() -> dict[str, str]:
    """Register collection tasks in collections folder.

    Each environment is named collection_name.env_name-v0
    """
    derived_envs = {}
    # TODO: Temporary disabling priors task
    collection_libs = ["perceptualdecisionmaking", "yang19"]
    for collection_lib in collection_libs:
        lib = f"neurogym.envs.collections.{collection_lib}"
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


def all_envs(
    tag: str | None = None,
    psychopy: bool = False,
    contrib: bool = False,
    collections: bool = False,
) -> list[str]:
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

    new_env_list: list[str] = []
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


def make(id_: str, **kwargs) -> gym.Env:
    """Creates an environment previously registered with :meth:`ngym.register`.

    This function calls the Gymnasium `make` function with the `disable_env_checker` argument set to True.

    Args:
        id_: A string representing the environment ID.
        kwargs: Additional arguments to pass to the environment constructor.

    Returns:
        An instance of the environment.
    """
    # built in env_checker raises warnings or errors when ob not in observation_space
    return gym.make(id_, disable_env_checker=True, **kwargs)


def register(id_: str, **kwargs) -> None:
    all_gym_envs = [env.id for env in gym.envs.registry.values()]
    if id_ not in all_gym_envs:
        gym.envs.registration.register(id=id_, **kwargs)


for env_id, entry_point in ALL_EXTENDED_ENVS.items():
    register(id_=env_id, entry_point=entry_point)
