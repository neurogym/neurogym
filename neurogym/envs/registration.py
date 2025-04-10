import importlib
from inspect import getmembers, isclass, isfunction
from pathlib import Path

import gymnasium as gym
from rapidfuzz import process

from neurogym.envs.collections import get_collection


def _get_envs(
    env_names: list[str],
    foldername: str | None = None,
    env_prefix: str | None = None,
) -> dict[str, str]:
    """A helper function to discover and register environments from a specified folder.

    This function scans Python files in the specified folder, imports them, and registers
    environments whose names match the provided `env_names` list.

    Example usage:
        _get_envs(env_names=NATIVE_ALLOW_LIST, foldername=None, env_prefix=None)
        _get_envs(env_names=CONTRIB_ALLOW_LIST, foldername='contrib', env_prefix='contrib')

    Args:
        env_names: A list of allowed environment names for manual curation.
        foldername: A string specifying the subfolder within `neurogym.envs` to search for
            environment files. If `None`, the function searches in the root folder of
            `neurogym.envs`. For example:
                - foldername=None: Searches in `neurogym/envs/`.
                - foldername="contrib": Searches in `neurogym/envs/contrib/`.
        env_prefix: A string to add as a prefix to all environment IDs. If provided, it
            ensures that the registered environment IDs are namespaced appropriately.

    Returns:
        A dictionary mapping environment IDs to their entry points.
    """
    if env_prefix is None:
        env_prefix = ""
    elif env_prefix[-1] != ".":
        env_prefix += "."

    # Root path of neurogym.envs folder
    env_root = Path(__file__).resolve().parent
    lib_root = "neurogym.envs."
    if foldername is not None:
        env_root /= foldername
        lib_root = f"{lib_root}{foldername}."

    # Only take .py files
    files = [p for p in env_root.iterdir() if p.suffix == ".py"]
    # Exclude files starting with '_'
    files = [f for f in files if f.name[0] != "_"]
    filenames = [f.name[:-3] for f in files]  # remove .py suffix
    filenames = sorted(filenames)

    env_dict: dict[str, str] = {}
    for filename in filenames:
        lib = lib_root + filename
        module = importlib.import_module(lib)
        for name, _val in getmembers(module):
            if name in env_names:
                env_dict[f"{env_prefix}{name}-v0"] = f"{lib}:{name}"

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

    This function attempts to create an environment using the given `id_`. If the environment
    is not registered, it raises an error and suggests the closest matching environment IDs.

    Args:
        id_: A string representing the environment ID.
        kwargs: Additional arguments to pass to the environment constructor.

    Returns:
        An instance of the environment.

    Raises:
        gym.error.UnregisteredEnv: If the `id_` doesn't exist in the Neurogym's registry. The error
            message will include suggestions for the closest matching environment IDs.
    """
    try:
        # built in env_checker raises warnings or errors when ob not in observation_space
        return gym.make(id_, disable_env_checker=True, **kwargs)

    except gym.error.UnregisteredEnv as e:
        raise _handle_unregistered_env_error(id_) from e


def _handle_unregistered_env_error(id_: str) -> gym.error.UnregisteredEnv:
    registered_envs = [env.id for env in gym.envs.registry.values()]
    env_guesses = _get_closest_matches(id_, registered_envs)

    error_msg = f"No registered env with id_: {id_}."
    if env_guesses:
        error_msg += "\nPerhaps you meant one of the following:\n"
        for guess in env_guesses:
            error_msg += f"    {guess}\n"

    return gym.error.UnregisteredEnv(error_msg)


def _get_closest_matches(
    id_: str,
    registered_envs: list[str],
    n_guesses: int = 5,
) -> list[str]:
    """Find the closest matching strings to a given ID from a list of IDs.

    This function uses fuzzy string matching to identify the closest matches to the provided `id_` from the
    `registered_envs` list. The number of matches returned is limited by `n_guesses`.

    Args:
        id_: The target string to match against the list of IDs.
        registered_envs: A list of strings to search for matches.
        n_guesses: The maximum number of closest matches to return.
            Defaults to 5.

    Returns:
        A list of the closest matching strings from `registered_envs`.
    """
    closest_matches = process.extract(
        id_,
        registered_envs,
        limit=n_guesses,
    )

    return [match for match, _, _ in closest_matches]


# backward compatibility with old versions of gym
# FIXME: check if backward compatibility is still required
def register(id_: str, **kwargs) -> None:
    if hasattr(gym.envs.registry, "all"):
        _all_gym_envs = [env.id for env in gym.envs.registry.all()]
    else:
        _all_gym_envs = [env.id for env in gym.envs.registry.values()]

    if id_ not in _all_gym_envs:
        gym.envs.registration.register(id=id_, **kwargs)


for env_id, entry_point in ALL_EXTENDED_ENVS.items():
    register(id_=env_id, entry_point=entry_point)
