import neurogym as ngym
from neurogym.core import TrialWrapper
import numpy as np


class TrialHistory(TrialWrapper):
    """Change ground truth probability based on previous outcome.

    Args:
        probs: matrix of probabilities of the current choice conditioned
            on the previous for each block. (def: None, np.array,
            num-blocks x num-choices x num-choices)
        num_blocks: if 2, repeating and ascending blocks created; if 3,
        an extra descending blocks is added
        block_dur: Number of trials per block. (def: 200 (int))
        blk_ch_prob: If not None, specifies the probability of changing block
            (randomly). (def: None, float)
    """
    metadata = {
        'description': 'Change ground truth probability based on previous' +
        'outcome.',
        'paper_link': 'https://www.biorxiv.org/content/10.1101/433409v3',
        'paper_name': 'Response outcomes gate the impact of expectations ' +
        'on perceptual decisions'
    }

    def __init__(self, env, probs=None, block_dur=200, num_blocks=2,
                 blk_ch_prob=None, rand_blcks=False, balanced_probs=True):
        super().__init__(env)
        try:
            self.n_ch = len(self.unwrapped.choices)  # max num of choices
            self.th_choices = self.unwrapped.choices
            self.curr_n_ch = self.n_ch
        except AttributeError:
            raise AttributeError('''SideBias requires task
                                 to have attribute choices''')
        assert isinstance(self.unwrapped, ngym.TrialEnv), 'Task has to be TrialEnv'
        assert probs is not None, 'Please provide choices probabilities'
        self.probs = probs
        self.balanced_probs = balanced_probs
        self.num_blocks = num_blocks
        self.rand_blcks = rand_blcks
        self.curr_tr_mat = self.trans_probs
        assert self.curr_tr_mat.shape[1] == self.n_ch,\
            'The number of choices {:d}'.format(self.tr_mat.shape[1]) +\
            ' inferred from prob mismatchs {:d}'.format(self.n_ch) +\
            ' inferred from choices'
        self.block_dur = block_dur
        self.prev_trial = self.rng.choice(self.n_ch)  # random initialization
        self.blk_ch_prob = blk_ch_prob

    def new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Periods
        # ---------------------------------------------------------------------
        block_already_changed = False
        # Check if n_ch is passed and if it is different from previous value
        if 'n_ch' in kwargs.keys() and kwargs['n_ch'] != self.curr_n_ch:
            self.curr_n_ch = kwargs['n_ch']
            self.prev_trial = self.rng.choice(self.th_choices[:self.curr_n_ch])
            self.curr_tr_mat = self.trans_probs
            block_already_changed = True

        # change rep. prob. every self.block_dur trials
        if not block_already_changed:
            if self.blk_ch_prob is None:
                block_change = self.unwrapped.num_tr % self.block_dur == 0
            else:
                block_change = self.unwrapped.rng.rand() < self.blk_ch_prob
            if block_change:
                if self.rand_blcks:
                    self.curr_tr_mat = self.trans_probs
                else:
                    self.curr_block = (self.curr_block + 1) % self.curr_n_blocks
                    self.blk_id = self.curr_block

        probs_curr_blk = self.curr_tr_mat[self.curr_block, self.prev_trial, :]
        ground_truth = self.unwrapped.rng.choice(self.th_choices[:self.curr_n_ch],
                                                 p=probs_curr_blk)
        self.prev_trial =\
            np.where(self.th_choices[:self.curr_n_ch] == ground_truth)[0][0]
        kwargs.update({'ground_truth': ground_truth,
                       'curr_block': self.curr_block})
        self.env.new_trial(**kwargs)

    @property
    def trans_probs(self):
        '''
        if prob is float it creates the transition matrix
        if prob is already a matrix it normalizes the probabilities and extracts
        the subset corresponding to the current number of choices
        '''
        if isinstance(self.probs, float):
            if self.rand_blcks:
                if self.balanced_probs:
                    indx = np.arange(self.curr_n_ch)
                    np.random.shuffle(indx)
                else:
                    indx = np.random.choice(self.curr_n_ch, size=(self.curr_n_ch,))
                tr_mat = np.eye(self.curr_n_ch)*self.probs
                tr_mat[tr_mat == 0] = (1-self.probs)/(self.curr_n_ch-1)
                tr_mat = tr_mat[indx, :]
                tr_mat = np.expand_dims(tr_mat, axis=0)
            else:
                tr_mat =\
                    np.zeros((self.num_blocks, self.curr_n_ch, self.curr_n_ch)) +\
                    (1-self.probs)/(self.curr_n_ch-1)
                for ind in range(self.curr_n_ch):
                    # ascending
                    tr_mat[0, ind, (ind+1) % self.curr_n_ch] = self.probs
                    # repeating block
                    tr_mat[1, ind, ind] = self.probs
                    if self.num_blocks == 3:
                        tr_mat[2, ind, ind-1] = self.probs  # descending block
        else:
            tr_mat = self.probs.copy()
            scaled_tr_mat = tr_mat[:, :self.curr_n_ch, :self.curr_n_ch]
            scaled_tr_mat /= np.sum(scaled_tr_mat, axis=2, keepdims=True)
            tr_mat = scaled_tr_mat
        tr_mat = np.unique(tr_mat, axis=0)
        self.curr_n_blocks = tr_mat.shape[0]
        self.curr_block = self.unwrapped.rng.choice(range(self.curr_n_blocks))
        self.blk_id = int(''.join([str(x+1) for x in indx])) if self.rand_blcks\
            else self.curr_block
        return tr_mat

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        info['curr_block'] = self.blk_id
        return obs, reward, done, info
