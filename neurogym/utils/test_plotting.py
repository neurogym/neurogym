import numpy as np

from neurogym.utils.plotting import fig_


obs = np.random.randn(20, 3)
actions = np.random.randint(0, 5, size=(20,))
gt = np.random.randint(0, 5, size=(20,))
rewards = np.random.randn(20)


fig_(obs, actions, gt, rewards)
