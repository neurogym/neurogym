import importlib
from inspect import getmembers, isfunction, isclass

import gym
from gym.envs.registration import register

from neurogym.envs.collections import get_collection

ALL_NATIVE_ENVS = {
    'ContextDecisionMaking-v0':
        'neurogym.envs.contextdecisionmaking:ContextDecisionMaking',
    'SingleContextDecisionMaking-v0':
        'neurogym.envs.contextdecisionmaking:SingleContextDecisionMaking',
    'DelayComparison-v0':
        'neurogym.envs.delaycomparison:DelayComparison',
    'PerceptualDecisionMaking-v0':
        'neurogym.envs.perceptualdecisionmaking:PerceptualDecisionMaking',
    'EconomicDecisionMaking-v0':
        'neurogym.envs.economicdecisionmaking:EconomicDecisionMaking',
    'PostDecisionWager-v0':
        'neurogym.envs.postdecisionwager:PostDecisionWager',
    'DelayPairedAssociation-v0':
        'neurogym.envs.delaypairedassociation:DelayPairedAssociation',
    'GoNogo-v0':
        'neurogym.envs.gonogo:GoNogo',
    'ReadySetGo-v0':
        'neurogym.envs.readysetgo:ReadySetGo',
    'OneTwoThreeGo-v0':
        'neurogym.envs.readysetgo:OneTwoThreeGo',
    'DelayMatchSample-v0':
        'neurogym.envs.delaymatchsample:DelayMatchSample',
    'DelayMatchCategory-v0':
        'neurogym.envs.delaymatchcategory:DelayMatchCategory',
    'DawTwoStep-v0':
        'neurogym.envs.dawtwostep:DawTwoStep',
    'HierarchicalReasoning-v0':
        'neurogym.envs.hierarchicalreasoning:HierarchicalReasoning',
    'MatchingPenny-v0':
        'neurogym.envs.matchingpenny:MatchingPenny',
    'MotorTiming-v0':
        'neurogym.envs.readysetgo:MotorTiming',
    'MultiSensoryIntegration-v0':
        'neurogym.envs.multisensory:MultiSensoryIntegration',
    'Bandit-v0':
        'neurogym.envs.bandit:Bandit',
    'PerceptualDecisionMakingDelayResponse-v0':
        'neurogym.envs.perceptualdecisionmaking:PerceptualDecisionMakingDelayResponse',
    'NAltPerceptualDecisionMaking-v0':
        'neurogym.envs.nalt_perceptualdecisionmaking:nalt_PerceptualDecisionMaking',
    # 'Combine-v0': 'neurogym.envs.combine:combine',
    # 'IBL-v0': 'neurogym.envs.ibl:IBL',
    # 'MemoryRecall-v0': 'neurogym.envs.memoryrecall:MemoryRecall',
    'Reaching1D-v0':
        'neurogym.envs.reaching:Reaching1D',
    'Reaching1DWithSelfDistraction-v0':
        'neurogym.envs.reaching:Reaching1DWithSelfDistraction',
    'AntiReach-v0':
        'neurogym.envs.antireach:AntiReach',
    'DelayMatchSampleDistractor1D-v0':
        'neurogym.envs.delaymatchsample:DelayMatchSampleDistractor1D',
    'IntervalDiscrimination-v0':
        'neurogym.envs.intervaldiscrimination:IntervalDiscrimination',
    'AngleReproduction-v0':
        'neurogym.envs.anglereproduction:AngleReproduction',
    'Detection-v0':
        'neurogym.envs.detection:Detection',
    'ReachingDelayResponse-v0':
        'neurogym.envs.reachingdelayresponse:ReachingDelayResponse',
    'CVLearning-v0':
        'neurogym.envs.cv_learning:CVLearning',
    'ChangingEnvironment-v0':
        'neurogym.envs.changingenvironment:ChangingEnvironment',
    'ProbabilisticReasoning-v0':
        'neurogym.envs.weatherprediction:ProbabilisticReasoning',
    'DualDelayMatchSample-v0':
        'neurogym.envs.dualdelaymatchsample:DualDelayMatchSample',
    'PulseDecisionMaking-v0':
        'neurogym.envs.perceptualdecisionmaking:PulseDecisionMaking',
    'Nothing-v0':
        'neurogym.envs.nothing:Nothing',
    'Pneumostomeopening-v0':
        'neurogym.envs.pneumostomeopening:Pneumostomeopening',
    'spatialsuppressmotion2-v0':
        'neurogym.envs.spatialsuppressmotion2:SpatialSuppressMotion2',
}

ALL_PSYCHOPY_ENVS = {
    'psychopy.RandomDotMotion-v0':
        'neurogym.envs.psychopy.perceptualdecisionmaking:RandomDotMotion',
    'psychopy.VisualSearch-v0':
        'neurogym.envs.psychopy.visualsearch:VisualSearch',
    'psychopy.spatialsuppressmotion-v0':
        'neurogym.envs.psychopy.spatialsuppressmotion:SpatialSuppressMotion',
}


# Automatically register all tasks in collections
def _get_collection_envs():
    """Register collection tasks in collections folder.

    Each environment is named collection_name.env_name-v0
    """
    derived_envs = {}
    collection_libs = ['perceptualdecisionmaking', 'yang19']
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
    **ALL_NATIVE_ENVS, **ALL_PSYCHOPY_ENVS
}

ALL_EXTENDED_ENVS = {**ALL_ENVS, **ALL_COLLECTIONS_ENVS}


def all_envs(tag=None, psychopy=False, collections=False):
    """Return a list of all envs in neurogym."""
    envs = ALL_NATIVE_ENVS.copy()
    if psychopy:
        envs.update(ALL_PSYCHOPY_ENVS)
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
