import neurogym as ngym
from neurogym.envs.annubes import AnnubesEnv

env = AnnubesEnv(fix_time=("until", 15000))
# env = AnnubesEnv(fix_time=500)
env.reset()
