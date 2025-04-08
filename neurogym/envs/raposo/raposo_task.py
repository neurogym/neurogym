import numpy as np


class RaposoTask:
    def __init__(self, task_param):
        self.fmin = task_param['fmin']
        self.fmax = task_param['fmax']
        self.modalities = ['v', 'a', 'va']
        self.freq_step = task_param['freq_step']
        self.freqs = np.arange(self.fmin, self.fmax + self.freq_step, self.freq_step)
        self.boundary = np.mean(self.freqs)
        self.nconditions = 3 * len(self.freqs)  # a trial can either be visual, auditory or multisensory
        self.use_fixation = task_param.get('fixation', False)

        # NOTE: I am using fixation as 100. Robert used fixation as 100 during training and 500 during testing
        # NOTE: Robert also used 300 units of time for 'decision' after 'stimulus'. I am not using it.
        self.fixation = 100
        self.stimulus = 1000  # stimulus is present for model second
        self.T = self.fixation + self.stimulus

        self.Nin = 5  # number of inputs
        self.Nout = 2  # number of outputs
        self.baseline_inp = 0.2  # baseline input for all neurons
        self.tau = task_param['tau']
        self.std_inp_noise = task_param['std_inp_noise']  # standard deviation for input noise
        # Input labels
        self.VISUAL_P = 0  # visual input greater than boundary frequency
        self.AUDITORY_P = 1  # auditory input greater than boundary frequency
        self.VISUAL_N = 2  # visual input lower than boundary frequency
        self.AUDITORY_N = 3  # auditory input lower than boundary frequency
        self.START = 4  # start cue

        # desired network outputs
        self.high_output = 1
        self.low_output = 0.2

    def scale_v_p(self, f):
        return 0.4 + 0.8 * (f - self.fmin) / (self.fmax - self.fmin)

    def scale_a_p(self, f):
        return 0.4 + 0.8 * (f - self.fmin) / (self.fmax - self.fmin)

    def scale_v_n(self, f):
        return 0.4 + 0.8 * (self.fmax - f) / (self.fmax - self.fmin)

    def scale_a_n(self, f):
        return 0.4 + 0.8 * (self.fmax - f) / (self.fmax - self.fmin)

    def generate_trials(self, rng, dt, minibatch_size):
        # -------------------------------------------------------------------------------------
        # Select task condition
        # -------------------------------------------------------------------------------------

        modality = rng.choice(self.modalities, minibatch_size)
        freq = rng.choice(self.freqs, minibatch_size)

        # -------------------------------------------------------------------------------------
        # Setup phases of trial
        # -------------------------------------------------------------------------------------

        t = np.linspace(dt, self.T, int(self.T / dt))
        phases = {}
        if self.use_fixation:
            phases['fixation'] = np.where(t <= self.fixation)[0]
        phases['stimulus'] = np.where(t > self.fixation)[0]

        # -------------------------------------------------------------------------------------
        # Trial Info
        # -------------------------------------------------------------------------------------

        choice = (freq > self.boundary).astype(np.int_)

        trials = {}
        trials['modality'] = modality
        trials['freq'] = freq
        trials['choice'] = choice
        trials['phases'] = phases

        # -------------------------------------------------------------------------------------
        # Inputs
        # -------------------------------------------------------------------------------------


        x = np.zeros((minibatch_size, len(t), self.Nin), dtype=np.float32)

        for i in range(minibatch_size):
            if 'v' in modality[i]:
                # input for visual trials
                x[i, phases['stimulus'], self.VISUAL_P] = self.scale_v_p(freq[i])
                x[i, phases['stimulus'], self.VISUAL_N] = self.scale_v_n(freq[i])

            if 'a' in modality[i]:
                # input for auditory trials
                x[i, phases['stimulus'], self.AUDITORY_P] = self.scale_a_p(freq[i])
                x[i, phases['stimulus'], self.AUDITORY_N] = self.scale_a_n(freq[i]) #!Same number, besides noise for 900ms

            x[i, phases['stimulus'], self.START] = 1

        # add noise to inputs
        alpha = dt/self.tau

        inp_noise = 1/alpha * np.sqrt(2 * alpha) * self.std_inp_noise * rng.normal(loc=0, scale=1, size=x.shape)

        trials['inputs'] = np.maximum(0,x + self.baseline_inp + inp_noise)


        # -------------------------------------------------------------------------------------
        # target output
        # -------------------------------------------------------------------------------------

        y = np.zeros((minibatch_size, len(t), self.Nout), dtype=np.float32)
        for i in range(minibatch_size):
            #Keep low during fixation
            if self.use_fixation:
                y[i, phases['fixation'], :] = self.low_output

            y[i, phases['stimulus'], choice[i]] = self.high_output
            y[i, phases['stimulus'], 1 - choice[i]] = self.low_output

        trials['outputs'] = y

        return trials
