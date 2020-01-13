import gym
from gym.envs.registration import register

from neurogym.core import BaseEnv
from neurogym.core import TrialEnv
from neurogym.core import EpochEnv


def register_neuroTask(id_task):
    for env in gym.envs.registry.all():
        if env.id == id_task:
            return
    register(id=id_task, entry_point=all_tasks[id_task])


all_tasks = {'Mante-v0': 'neurogym.envs.mante:Mante',
             'Romo-v0': 'neurogym.envs.romo:Romo',
             'RDM-v0': 'neurogym.envs.rdm:RDM',
             'padoaSch-v0': 'neurogym.envs.padoa_sch:PadoaSch',
             'pdWager-v0': 'neurogym.envs.pd_wager:PDWager',
             'DPA-v0': 'neurogym.envs.dpa:DPA',
             'GNG-v0': 'neurogym.envs.gng:GNG',
             'ReadySetGo-v0': 'neurogym.envs.readysetgo:ReadySetGo',
             'DelayedMatchSample-v0':
                 'neurogym.envs.delaymatchsample:DelayedMatchToSample',
             'DelayedMatchCategory-v0':
                 'neurogym.envs.delaymatchcategory:DelayedMatchCategory',
             'DawTwoStep-v0': 'neurogym.envs.dawtwostep:DawTwoStep',
             'MatchingPenny-v0': 'neurogym.envs.matchingpenny:MatchingPenny',
             'Bandit-v0': 'neurogym.envs.bandit:Bandit',
             'DelayedResponse-v0': 'neurogym.envs.delayresponse:DR',
             'NAltRDM-v0': 'neurogym.envs.nalt_rdm:nalt_RDM',
             'GenTask-v0': 'neurogym.envs.generaltask:GenTask',
             'Combine-v0': 'neurogym.envs.combine:combine',
             'IBL-v0': 'neurogym.envs.ibl:IBL',
             'MemoryRecall-v0': 'neurogym.envs.memoryrecall:MemoryRecall',
             }

for task in all_tasks.keys():
    register_neuroTask(task)
