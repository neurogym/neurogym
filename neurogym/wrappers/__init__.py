from neurogym.wrappers.block import RandomGroundTruth, ScheduleAttr, ScheduleEnvs, TrialHistoryV2
from neurogym.wrappers.monitor import Monitor
from neurogym.wrappers.noise import Noise
from neurogym.wrappers.pass_action import PassAction
from neurogym.wrappers.pass_reward import PassReward
from neurogym.wrappers.reaction_time import ReactionTime
from neurogym.wrappers.side_bias import SideBias

ALL_WRAPPERS = {
    "Monitor-v0": "neurogym.wrappers.monitor:Monitor",
    "Noise-v0": "neurogym.wrappers.noise:Noise",
    "PassReward-v0": "neurogym.wrappers.pass_reward:PassReward",
    "PassAction-v0": "neurogym.wrappers.pass_action:PassAction",
    "ReactionTime-v0": "neurogym.wrappers.reaction_time:ReactionTime",
    "SideBias-v0": "neurogym.wrappers.side_bias:SideBias",
    "RandomGroundTruth-v0": "neurogym.wrappers.block:RandomGroundTruth",
    "ScheduleAttr-v0": "neurogym.wrappers.block:ScheduleAttr",
    "ScheduleEnvs-v0": "neurogym.wrappers.block:ScheduleEnvs",
    "TrialHistoryV2-v0": "neurogym.wrappers.block:TrialHistoryV2",
}


def all_wrappers():
    return sorted(ALL_WRAPPERS)
