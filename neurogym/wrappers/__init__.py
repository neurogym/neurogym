from neurogym.wrappers.catch_trials import CatchTrials
from neurogym.wrappers.monitor import Monitor
from neurogym.wrappers.noise import Noise
from neurogym.wrappers.pass_reward import PassReward
from neurogym.wrappers.miss_trials_reward import MissTrialReward
from neurogym.wrappers.pass_action import PassAction
from neurogym.wrappers.reaction_time import ReactionTime
from neurogym.wrappers.side_bias import SideBias
from neurogym.wrappers.trial_hist import TrialHistory
from neurogym.wrappers.ttl_pulse import TTLPulse
from neurogym.wrappers.combine import Combine
from neurogym.wrappers.identity import Identity
from neurogym.wrappers.transfer_learning import TransferLearning


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
                    'neurogym.wrappers.ttl_pulse:TTLPulse',
                'Combine-v0':
                    'neurogym.wrappers.combine:Combine',
                'Identity-v0':
                    'neurogym.wrappers.identity:Identity',
                'TransferLearning-v0':
                    'neurogym.wrappers.transfer_learning:TransferLearning'
                }


def all_wrappers():
    return sorted(list(ALL_WRAPPERS.keys()))
