import neurogym as ngym
import numpy as np


class TrialHistory(ngym.TrialWrapper):
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
                 blk_ch_prob=None):
        super().__init__(env)
        try:
            self.n_ch = len(self.task.choices)  # max num of choices
            self.th_choices = self.task.choices
            self.curr_n_ch = self.n_ch
        except AttributeError:
            raise AttributeError('''SideBias requires task
                                 to have attribute choices''')
        assert isinstance(self.task, ngym.TrialEnv), 'Task has to be TrialEnv'
        assert probs is not None, 'Please provide choices probabilities'
        self.probs = probs
        self.num_blocks = num_blocks
        self.curr_tr_mat = self.trans_probs
        assert self.curr_tr_mat.shape[1] == self.n_ch,\
            'The number of choices {:d}'.format(self.tr_mat.shape[1]) +\
            ' inferred from prob mismatchs {:d}'.format(self.n_ch) +\
            ' inferred from choices'
        self.n_block = self.curr_tr_mat.shape[0]
        self.curr_block = self.task.rng.choice(range(self.n_block))
        self.block_dur = block_dur
        self.prev_trial = self.rng.choice(self.n_ch)  # random initialization
        self.blk_ch_prob = blk_ch_prob

    def new_trial(self, **kwargs):
        # ---------------------------------------------------------------------
        # Periods
        # ---------------------------------------------------------------------
        # change rep. prob. every self.block_dur trials
        if self.blk_ch_prob is None:
            if self.task.num_tr % self.block_dur == 0:
                self.curr_block = (self.curr_block + 1) % self.n_block
        else:
            if self.task.rng.random() < self.blk_ch_prob:
                self.curr_block = (self.curr_block + 1) % self.n_block

        # Check if n_ch is passed and if it is different from previous value
        if 'n_ch' in kwargs.keys() and kwargs['n_ch'] != self.curr_n_ch:
            self.curr_n_ch = kwargs['n_ch']
            self.prev_trial = self.rng.choice(self.th_choices[:self.curr_n_ch])
            self.curr_tr_mat = self.trans_probs

        probs_curr_blk = self.curr_tr_mat[self.curr_block, self.prev_trial, :]
        ground_truth = self.task.rng.choice(self.th_choices[:self.curr_n_ch],
                                            p=probs_curr_blk)
        self.prev_trial =\
            np.where(self.th_choices[:self.curr_n_ch] == ground_truth)[0][0]
        kwargs.update({'ground_truth': ground_truth,
                       'curr_block': self.curr_block})
        self.env.new_trial(**kwargs)

    @property
    def trans_probs(self):
        '''
        Creates transition matrix if prob is float or if prob is already
        the matrix it normalizes the probabilities and extracts a subset.
        '''
        if isinstance(self.probs, float):
            num_blocks = self.num_blocks
            tr_mat =\
                np.zeros((num_blocks, self.curr_n_ch, self.curr_n_ch)) +\
                (1-self.probs)/(self.curr_n_ch-1)
            for ind in range(self.curr_n_ch):
                tr_mat[0, ind, (ind+1) % self.curr_n_ch] = self.probs  # ascending
                tr_mat[1, ind, ind] = self.probs    # repeating block
                if num_blocks == 3:
                    tr_mat[2, ind, ind-1] = self.probs  # descending block
        else:
            tr_mat = self.probs.copy()
            scaled_tr_mat = tr_mat[:, :self.curr_n_ch, :self.curr_n_ch]
            scaled_tr_mat /= np.sum(scaled_tr_mat, axis=2, keepdims=True)
            tr_mat = scaled_tr_mat
        return tr_mat

    def step(self, action, new_tr_fn=None):
        ntr_fn = new_tr_fn or self.new_trial
        obs, reward, done, info = self.env.step(action, new_tr_fn=ntr_fn)
        info['curr_block'] = self.curr_block
        return obs, reward, done, info
