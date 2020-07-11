import numpy as np
from gym import spaces
import neurogym as ngym

# XXX: are trials counted correctly in this task?

# TODO: No longer maintained, use existing task to build this task


class IBL(ngym.TrialEnv):
    metadata = {
        'paper_link': 'https://www.sciencedirect.com/science/article/' +
        'pii/S0896627317311364',
        'paper_name': '''An International Laboratory for Systems and ' +
        'Computational Neuroscience''',
    }

    def __init__(self, dt=100, rewards=None):
        super(IBL, self).__init__(dt=dt)
        # TODO: Fix to use the default random number generator
        self._rng = self.rng.RandomState(0)
        self.sigma = 0.10  # noise
        self.num_tr = 0  # number of trials
        self.block = 0  # block id
        self.block_size = 10000

        # Rewards
        self.rewards = {'correct': +1, 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        # trial conditions (left, right)
        self.choices = [0, 1]

        self.cohs = np.array([1.6, 3.2, 6.4, 12.8, 25.6, 51.2])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(2,),
                                            dtype=np.float32)

    def new_block(self, n_trial, probs=None):
        self.ground_truth = self._rng.choice(self.choices,
                                             size=(n_trial,), p=probs)
        self.coh = self._rng.choice(self.cohs, size=(n_trial,))

        obs = np.zeros((n_trial, self.observation_space.shape[0]))
        ind = np.arange(n_trial)
        obs[ind, self.ground_truth] = 0.5 + self.coh / 200
        obs[ind, 1 - self.ground_truth] = 0.5 - self.coh / 200

        # Add observation noise
        obs += self._rng.randn(*obs.shape) * self.sigma
        self.ob = obs

    def _new_trial(self, **kwargs):
        """
        _new_trial() is called when a trial ends to get the specifications of
        the next trial. Such specifications are stored in a dictionary with
        the following items:
            durations, which stores the duration of the different periods (in
            the case of perceptualDecisionMaking: fixation, stimulus and decision periods)
            ground truth: correct response for the trial
            coh: stimulus coherence (evidence) for the trial
        """
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        self.ind = self.num_tr % self.block_size
        if self.ind == 0:
            self.new_block(self.block_size)

        self.num_tr += 1

    def _step(self, action):
        info = {'continue': True, 'gt': self.ground_truth[self.ind],
                'coh': self.coh[self.ind], 'block': self.block}
        obs = self.ob[self.ind]

        # reward of last trial
        reward = self.rewards['correct']  # TODO: need to be done

        # ---------------------------------------------------------------------
        # new trial?
        info['new_trial'] = True
        done = False
        return obs, reward, done, info


class IBL_Block(IBL):
    # pass
    def __init__(self, dt=100):
        super().__init__(dt=dt)
        self.probs = ((0.2, 0.8), (0.8, 0.2), (0.5, 0.5))
        self.block = 0
        self.block_size = 200

    def _new_trial(self, **kwargs):
        """
        _new_trial() is called when a trial ends to get the specifications of
        the next trial. Such specifications are stored in a dictionary with
        the following items:
            durations, which stores the duration of the different periods (in
            the case of perceptualDecisionMaking: fixation, stimulus and decision periods)
            ground truth: correct response for the trial
            coh: stimulus coherence (evidence) for the trial
        """
        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------
        self.ind = self.num_tr % self.block_size
        if self.ind == 0:
            self.block = self._rng.choice([0, 1, 2])
            prob = self.probs[self.block]
            self.new_block(self.block_size, probs=prob)

        self.num_tr += 1
