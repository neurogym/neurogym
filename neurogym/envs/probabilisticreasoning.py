"""Random dot motion task."""

import numpy as np

import neurogym as ngym
from neurogym import spaces


class ProbabilisticReasoning(ngym.TrialEnv):
    """Probabilistic reasoning.

    The agent is shown a sequence of stimuli. Each stimulus is associated
    with a certain log-likelihood of the correct response being one choice
    versus the other. The final log-likelihood of the target response being,
    for example, option 1, is the sum of all log-likelihood associated with
    the presented stimuli. A delay period separates each stimulus, so the
    agent is encouraged to lean the log-likelihood association and integrate
    these values over time within a trial.

    Args:
        shape_weight: array-like, evidence weight of each shape
        n_loc: int, number of location of show shapes
    """
    metadata = {
        'paper_link': 'https://www.nature.com/articles/nature05852',
        'paper_name': 'Probabilistic reasoning by neurons',
        'tags': ['perceptual', 'two-alternative', 'supervised']
    }

    def __init__(self, dt=100, rewards=None, timing=None, shape_weight=None,
                 n_loc=4):
        super().__init__(dt=dt)
        # The evidence weight of each stimulus
        if shape_weight is not None:
            self.shape_weight = shape_weight
        else:
            self.shape_weight = [-10, -0.9, -0.7, -0.5, -0.3,
                                 0.3, 0.5, 0.7, 0.9, 10]
            
        self.n_shape = len(self.shape_weight)
        dim_shape = self.n_shape
        # Shape representation needs to be fixed cross-platform
        self.shapes = np.eye(self.n_shape, dim_shape)
        self.n_loc = n_loc

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {'fixation': 500,
                       'delay': lambda: self.rng.uniform(450, 550),
                       'decision': 500
                       }
        for i_loc in range(n_loc):
            self.timing['stimulus'+str(i_loc)] = 500
        if timing:
            self.timing.update(timing)

        self.abort = False

        name = {'fixation': 0}
        start = 1
        for i_loc in range(n_loc):
            name['loc' + str(i_loc)] = range(start, start + dim_shape)
            start += dim_shape
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1 + dim_shape*n_loc,),
            dtype=np.float32, name=name)

        name = {'fixation': 0, 'choice': [1, 2]}
        self.action_space = spaces.Discrete(3, name=name)

    def _new_trial(self, **kwargs):
        # Trial info
        trial = {
            'locs': self.rng.choice(range(self.n_loc),
                                    size=self.n_loc, replace=False),
            'shapes': self.rng.choice(range(self.n_shape),
                                      size=self.n_loc, replace=True),
        }
        trial.update(kwargs)

        locs = trial['locs']
        shapes = trial['shapes']
        log_odd = sum([self.shape_weight[shape] for shape in shapes])
        p = 1. / (10**(-log_odd) + 1.)
        ground_truth = int(self.rng.rand() < p)
        trial['log_odd'] = log_odd
        trial['ground_truth'] = ground_truth

        # Periods
        periods = ['fixation']
        periods += ['stimulus'+str(i) for i in range(self.n_loc)]
        periods += ['delay', 'decision']
        self.add_period(periods)

        # Observations
        self.add_ob(1, where='fixation')
        self.set_ob(0, 'decision', where='fixation')

        for i_loc in range(self.n_loc):
            loc = locs[i_loc]
            shape = shapes[i_loc]
            periods = ['stimulus'+str(j) for j in range(i_loc, self.n_loc)]
            self.add_ob(self.shapes[shape], periods, where='loc'+str(loc))

        # Ground truth
        self.set_groundtruth(ground_truth, period='decision', where='choice')

        return trial

    def _step(self, action):
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        # observations
        if self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward += self.rewards['correct']
                    self.performance = 1
                else:
                    reward += self.rewards['fail']
        else:
            if action != 0:  # action = 0 means fixating
                new_trial = self.abort
                reward += self.rewards['abort']

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}
