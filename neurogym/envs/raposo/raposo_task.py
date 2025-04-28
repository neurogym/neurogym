import numpy as np
from gymnasium import spaces  # or: from gym import spaces
from neurogym.core import TrialEnv
from neurogym.envs.registration import register

class RaposoTask(TrialEnv):
    metadata = {'render.modes': ['human']}

    def __init__(self, task_param=None):
        if task_param is None:
            # Default params
            task_param = {
                'fmin': 9,
                'fmax': 16,
                'freq_step': 1,
                'std_inp_noise': 0.01,
                'fixation': True,
                'tau': 100,
            }
        self.fmin = task_param['fmin']
        self.fmax = task_param['fmax']
        self.freq_step = task_param['freq_step']
        self.freqs = np.arange(self.fmin, self.fmax + self.freq_step, self.freq_step)
        self.boundary = np.mean(self.freqs)
        self.modalities = ['v', 'a', 'va']
        self.choices = [0, 1]
        self.use_fixation = task_param.get('fixation', False)
        self.fixation = 100
        self.stimulus = 1000
        self.T = self.fixation + self.stimulus
        self.Nin = 5
        self.Nout = 2
        self.baseline_inp = 0.2
        self.tau = task_param['tau']
        self.std_inp_noise = task_param['std_inp_noise']
        self.high_output = 1
        self.low_output = 0.2

        self.observation_space = spaces.Box(low=0, high=1, shape=(self.Nin,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.Nout)

        super().__init__()

    def _new_trial(self, **kwargs):
        # Generate a new trial for RL
        trial = {}

        # Randomly pick trial parameters
        modality = self.rng.choice(self.modalities)
        freq = self.rng.choice(self.freqs)
        choice = int(freq > self.boundary)
        t = 0

        # Build input
        x = np.zeros(self.Nin, dtype=np.float32)
        if 'v' in modality:
            x[0] = self.scale_v_p(freq)
            x[2] = self.scale_v_n(freq)
        if 'a' in modality:
            x[1] = self.scale_a_p(freq)
            x[3] = self.scale_a_n(freq)
        x[4] = 1  # Start cue

        # Add noise
        alpha = 1 / self.tau
        inp_noise = 1 / alpha * np.sqrt(2 * alpha) * self.std_inp_noise * self.rng.normal(0, 1, x.shape)
        obs = np.maximum(0, x + self.baseline_inp + inp_noise)

        trial['observation'] = obs
        trial['correct_choice'] = choice
        trial['modality'] = modality
        trial['freq'] = freq
        trial['action'] = None  # Agent will act
        trial['reward'] = 0
        trial['done'] = False
        self.trial_data = trial
        return trial

    def _step(self, action):
        # Called by the agent/environment on every step
        correct = self.trial_data['correct_choice']
        reward = 1.0 if action == correct else 0.0
        self.trial_data['reward'] = reward
        self.trial_data['action'] = action
        self.trial_data['done'] = True
        return self.trial_data['observation'], reward, True, {}  # obs, reward, done, info

    def reset(self):
        trial = self._new_trial()
        return trial['observation']

    def scale_v_p(self, f):
        return 0.4 + 0.8 * (f - self.fmin) / (self.fmax - self.fmin)
    def scale_a_p(self, f):
        return 0.4 + 0.8 * (f - self.fmin) / (self.fmax - self.fmin)
    def scale_v_n(self, f):
        return 0.4 + 0.8 * (self.fmax - f) / (self.fmax - self.fmin)
    def scale_a_n(self, f):
        return 0.4 + 0.8 * (self.fmax - f) / (self.fmax - self.fmin)

# --- Register your environment ---
register(
    id_='CustomRaposoTask-v0',
    entry_point='neurogym.envs.raposo.raposo_task:RaposoTask',
)
