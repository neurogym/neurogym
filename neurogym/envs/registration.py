import importlib
from inspect import getmembers, isclass, isfunction
from pathlib import Path
from tkinter import ALL

import gymnasium as gym
from rapidfuzz import process

from neurogym.envs.collections import get_collection

try:
    import psychopy  # noqa: F401

    _have_psychopy = True  # FIXME should psychopy be always tested, to ensure CI doesn't fail?
except ImportError:
    _have_psychopy = False

EXCLUDE_ENVS = [
    # NATIVE_ENVS
    "Detection",
    "SpatialSuppressMotion",
    "ToneDetection",
    # CONTRIB_ENVS
    "AngleReproduction",
    "CVLearning",
    "ChangingEnvironment",
    "IBL",
    "MatchingPenny",
    "MemoryRecall",
    "Pneumostomeopening",
]

NATIVE_ENVS = [
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
    # "SpatialSuppressMotion",  # noqa: ERA001
    # TODO: raises ModuleNotFound error since requires scipy, which is not in the requirements of neurogym.
    # FIXME: I have added scipy to requirements (for other reason), does this mean SpatialSuppressMotion is valid?
    # "ToneDetection",  # TODO: Temporary removing until bug fixed # noqa: ERA001
]

_PSYCHOPY_PREFIX = "neurogym.envs.psychopy."
PSYCHOPY_ENVS = [
    "RandomDotMotion",
    "VisualSearch",
    "SpatialSuppressMotion",  # NOTE: this is different from the SpatialSuppressMotion native env. Consider renaming.
]

_CONTRIB_NAME_PREFIX = "contrib."
_CONTRIB_PREFIX = "neurogym.envs.contrib."
CONTRIB_ENVS: list[str] = [  # FIXME: why are these commented out? NOTE: mypy requires type hint for empty list
    # "AngleReproduction",
    # "CVLearning",
    # "ChangingEnvironment",
    # "IBL",
    # "MatchingPenny",
    # "MemoryRecall",
    # "Pneumostomeopening",
]


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
    # Root path of neurogym.envs folder
    env_root = Path(__file__).resolve().parent
    lib_root = "neurogym.envs."
    if foldername is not None:
        env_root /= foldername
        lib_root += f"{foldername}."

    if exclude is None:
        exclude = []
    elif isinstance(exclude, str):
        exclude = [exclude]
    elif not isinstance(exclude, list):
        msg = f"{type(exclude)=} must be a string or a list of strings."
        raise TypeError(msg)

    # Only take .py files
    files = [p for p in env_root.iterdir() if p.suffix == ".py" and p.name != "__init__.py"]

    env_dict: dict[str, str] = {}
    for file in files:
        module_name = lib_root + file.stem  # Convert filename to module name
        module = importlib.import_module(module_name)

        # Discover all classes in the module that inherit from gym.Env
        for name, obj in getmembers(module, isclass):
            if issubclass(obj, gym.Env) and obj.__module__ == module_name and name not in exclude:
                env_id = f"{foldername}.{name}-v0" if foldername else f"{name}-v0"
                entry_point = f"{module_name}:{name}"
                env_dict[env_id] = entry_point

    return env_dict


ALL_NATIVE_ENVS = _get_envs()
ALL_CONTRIB_ENVS = _get_envs("contrib")
try:
    ALL_PSYCHOPY_ENVS = _get_envs("psychopy")
except ModuleNotFoundError:
    ALL_PSYCHOPY_ENVS = {}
ALL_COLLECTIONS_ENVS = _get_envs("collections")

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
