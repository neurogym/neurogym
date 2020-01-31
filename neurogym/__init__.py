import gym
from gym.envs.registration import register

from neurogym.version import VERSION as __version__
from neurogym.core import BaseEnv
from neurogym.core import TrialEnv
from neurogym.core import PeriodEnv
from neurogym.core import TrialWrapper


ALL_TASKS = {'ContextDecisionMaking-v0': 'neurogym.envs.contextdecisionmaking:ContextDecisionMaking',
             'DelayedComparison-v0': 'neurogym.envs.delayedcomparison:DelayedComparison',
             'PerceptualDecisionMaking-v0':
                 'neurogym.envs.perceptualdecisionmaking:PerceptualDecisionMaking',
             'EconomicDecisionMaking-v0': 'neurogym.envs.economicdecisionmaking:EconomicDecisionMaking',
             'PostDecisionWager-v0': 'neurogym.envs.postdecisionwager:PostDecisionWager',
             'DelayPairedAssociation-v0':
                 'neurogym.envs.delaypairedassociation:DelayPairedAssociation',
             'GoNogo-v0': 'neurogym.envs.gonogo:GoNogo',
             'ReadySetGo-v0': 'neurogym.envs.readysetgo:ReadySetGo',
             'DelayedMatchSample-v0':
                 'neurogym.envs.delaymatchsample:DelayedMatchToSample',
             'DelayedMatchCategory-v0':
                 'neurogym.envs.delaymatchcategory:DelayedMatchCategory',
             'DawTwoStep-v0': 'neurogym.envs.dawtwostep:DawTwoStep',
             'MatchingPenny-v0': 'neurogym.envs.matchingpenny:MatchingPenny',
             'MotorTiming-v0': 'neurogym.envs.readysetgo:MotorTiming',
             'Bandit-v0': 'neurogym.envs.bandit:Bandit',
             'PerceptualDecisionMakingDelayResponse-v0':
                 'neurogym.envs.perceptualdecisionmaking:PerceptualDecisionMakingDelayResponse',
             'NAltPerceptualDecisionMaking-v0':
                 'neurogym.envs.nalt_perceptualdecisionmaking:nalt_PerceptualDecisionMaking',
             # 'Combine-v0': 'neurogym.envs.combine:combine',
             # 'IBL-v0': 'neurogym.envs.ibl:IBL',
             # 'MemoryRecall-v0': 'neurogym.envs.memoryrecall:MemoryRecall',
             'Reaching1D-v0': 'neurogym.envs.reaching:Reaching1D',
             'Reaching1DWithSelfDistraction-v0':
                 'neurogym.envs.reaching:Reaching1DWithSelfDistraction',
             'AntiReach-v0': 'neurogym.envs.antireach:AntiReach1D',
             'DelayedMatchToSampleDistractor1D-v0':
                 'neurogym.envs.delaymatchsample:DelayedMatchToSampleDistractor1D',
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
                 'neurogym.envs.changingenvironment:ChangingEnvironment'
             }


ALL_WRAPPERS = {'CatchTrials-v0': 'neurogym.wrappers.catch_trials:CatchTrials',
                'Monitor-v0': 'neurogym.wrappers.monitor:Monitor',
                'Noise-v0': 'neurogym.wrappers.noise:Noise',
                'PassReward-v0': 'neurogym.wrappers.pass_reward:PassReward',
                'PassAction-v0': 'neurogym.wrappers.pass_action:PassAction',
                'ReactionTime-v0':
                    'neurogym.wrappers.reaction_time:ReactionTime',
                'SideBias-v0': 'neurogym.wrappers.side_bias:SideBias',
                'TrialHistory-v0': 'neurogym.wrappers.trial_hist:TrialHistory',
                'MissTrialReward-v0':
                    'neurogym.wrappers.miss_trials_reward:MissTrialReward',
                'TTLPulse-v0':
                    'neurogym.wrappers.ttl_pulse:TTLPulse'
                }


def all_tasks():
    return ALL_TASKS.copy()


def all_wrappers():
    return ALL_WRAPPERS.copy()


def register_task(id_task):
    for env in gym.envs.registry.all():
        if env.id == id_task:
            return
    register(id=id_task, entry_point=ALL_TASKS[id_task])


for task in ALL_TASKS.keys():
    register_task(task)
