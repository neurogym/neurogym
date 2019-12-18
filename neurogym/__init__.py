# import sys
# from os.path import expanduser
# home = expanduser("~")
# sys.path.append(home)
# sys.path.append(home + '/gym')
import gym
from gym.envs.registration import register


def register_neuroTask(id_task):
    for env in gym.envs.registry.all():
        if env.id == id_task:
            return
    register(id=id_task, entry_point=all_tasks[id_task])


all_tasks = {'Mante-v0': 'neurogym.envs.mante:Mante',
             'Romo-v0': 'neurogym.envs.romo:Romo',
             'RDM-v0': 'neurogym.envs.rdm:RDM',
             'RDM-v1': 'neurogym.envs.rdm_v1:RDM',
             'padoaSch-v0': 'neurogym.envs.padoa_sch:PadoaSch',
             'pdWager-v0': 'neurogym.envs.pd_wager:PDWager',
             'DPA-v0': 'neurogym.envs.dpa:DPA',
             'GNG-v0': 'neurogym.envs.gng:GNG',
             'ReadySetGo-v0': 'neurogym.envs.readysetgo:ReadySetGo',
             'DelayedMatchSample-v0':
                 'neurogym.envs.delaymatchsample:DelayedMatchToSample',
             'DawTwoStep-v0': 'neurogym.envs.dawtwostep:DawTwoStep',
             'MatchingPenny-v0': 'neurogym.envs.matchingpenny:MatchingPenny',
             'Bandit-v0': 'neurogym.envs.bandit:Bandit',
             'DelayedResponse-v0': 'neurogym.envs.delayresponse:DR',
             'NAltRDM-v0': 'neurogym.envs.nalt_rdm:nalt_RDM',
             '2AFC-v0': 'neurogym.envs.2AFC:TwoAFC'}
for task in all_tasks.keys():
    register_neuroTask(task)
