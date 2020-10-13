from neurogym.wrappers.monitor import Monitor
from neurogym.wrappers.noise import Noise
from neurogym.wrappers.pass_reward import PassReward
from neurogym.wrappers.pass_action import PassAction
from neurogym.wrappers.reaction_time import ReactionTime
from neurogym.wrappers.side_bias import SideBias
from neurogym.wrappers.block import RandomGroundTruth
from neurogym.wrappers.block import ScheduleAttr
from neurogym.wrappers.block import ScheduleEnvs
from neurogym.wrappers.block import TrialHistoryV2

ALL_WRAPPERS = {'Monitor-v0': 'neurogym.wrappers.monitor:Monitor',
                'Noise-v0': 'neurogym.wrappers.noise:Noise',
                'PassReward-v0': 'neurogym.wrappers.pass_reward:PassReward',
                'PassAction-v0': 'neurogym.wrappers.pass_action:PassAction',
                'ReactionTime-v0':
                    'neurogym.wrappers.reaction_time:ReactionTime',
                'SideBias-v0': 'neurogym.wrappers.side_bias:SideBias',
                }

def all_wrappers():
    return sorted(list(ALL_WRAPPERS.keys()))
