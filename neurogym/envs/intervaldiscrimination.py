
from __future__ import division

import numpy as np
from gym import spaces
from neurogym.meta import info
import neurogym as ngym


# TODO: Getting duration is not intuitive, not clear to people
class IntervalDiscrimination(ngym.PeriodEnv):
    metadata = {
        'description': 'Agents have to report which of two stimuli presented' +
        ' sequentially is longer.',
        'paper_link': 'https://www.sciencedirect.com/science/article/pii/' +
        'S0896627309004887',
        'paper_name': """Feature- and Order-Based Timing Representations
         in the Frontal Cortex""",
        'timing': {  # TODO: Timing not from paper yet
            'fixation': ('constant', 300),
            'stim1': ('uniform', (300, 600)),
            'delay1': ('choice', [800, 1500]),
            'stim2': ('uniform', (300, 600)),
            'delay2': ('constant', 500),
            'decision': ('constant', 300)},
        'tags': ['timing', 'working memory', 'delayed response',
                 'two-alternative', 'supervised']
    }

    def __init__(self, dt=80, timing=None):
        super().__init__(dt=dt, timing=timing)
        # Rewards
        self.R_ABORTED = -0.1
        self.R_CORRECT = +1.
        self.R_FAIL = 0.
        self.abort = False
        # set action and observation space
        self.action_space = spaces.Discrete(3)  # (fixate, choose 1, choose2)
        # (fixation, stim1, stim2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,),
                                            dtype=np.float32)

    def new_trial(self, **kwargs):
        duration1 = self.timing_fn['stim1']()
        duration2 = self.timing_fn['stim2']()
        ground_truth = 1 if duration1 < duration2 else 2

        periods = ['fixation', 'stim1', 'delay1', 'stim2', 'delay2', 'decision']
        self.add_period(periods[0], after=0)
        for i in range(1, len(periods)):
            self.add_period(periods[i], after=periods[i - 1],
                           last_period=i == len(periods) - 1)

        self.set_ob('fixation', [1, 0, 0])
        self.set_ob('stim1', [1, 1, 0])
        self.set_ob('delay1', [1, 0, 0])
        self.set_ob('stim2', [1, 0, 1])
        self.set_ob('delay2', [1, 0, 0])
        self.set_ob('decision', [0, 0, 0])

        self.set_groundtruth('decision', ground_truth)

    def _step(self, action):
        # ---------------------------------------------------------------------
        # Reward and inputs
        # ---------------------------------------------------------------------
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        # observations
        if self.in_period('fixation'):
            if action != 0:  # action = 0 means fixating
                new_trial = self.abort
                reward = self.R_ABORTED
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward = self.R_CORRECT
                else:
                    reward = self.R_FAIL

        return self.obs_now, reward, False, {'new_trial': new_trial, 'gt': gt}


if __name__ == '__main__':
    from neurogym.tests import test_run
    env = IntervalDiscrimination()
    test_run(env)
    info.plot_struct(env)
