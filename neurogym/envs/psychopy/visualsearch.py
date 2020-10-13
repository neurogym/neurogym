"""Visual search task."""

import numpy as np
from gym import spaces
from psychopy import visual

from .psychopy_env import PsychopyEnv


class VisualSearch(PsychopyEnv):
    """Visual search task.

    Args:
        target_centers: list of tuples, centers of target stimuli
        length: float, length of target stimuli
        delta_angle: float, diff. between sample and distractor angles
        delta_color: 3-tuple, diff. between sample and distractor RGB color
        line_width: float, line width
    """
    metadata = {
        'paper_link': 'https://science.sciencemag.org/content/315/5820/1860',
        'paper_name': '''Top-down versus bottom-up control of attention 
        in the prefrontal and posterior parietal cortices''',
        'tags': ['perceptual', 'supervised']
    }

    def __init__(self, dt=16, win_kwargs={'size': (100, 100)}, rewards=None, timing=None,
                 target_centers=None, length=0.3, delta_angle=None,
                 delta_color=None, line_width=3):
        super().__init__(dt=dt, win_kwargs=win_kwargs)

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        self.timing = {
            'fixation': 500,
            'sample': 1000,
            'delay': 500,
            'decision': 500}
        if timing:
            self.timing.update(timing)

        # Task parameters
        if target_centers is None:
            self.target_centers = [(.5, .5), (.5, -.5), (-.5, .5), (-.5, -.5)]
        else:
            self.target_centers = target_centers
        self.length = length  # length
        self.delta_angle = delta_angle or 0.2 * np.pi
        self.delta_color = delta_color or [0.2, -0.1, -0.1]
        self.n_target = len(self.target_centers)  # number of target stimuli
        self.lw = line_width

        self.abort = False

        self.action_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)

    @staticmethod
    def _line_startend(center, angle, length):
        start = (center[0] - length * np.cos(angle),
                 center[1] - length * np.sin(angle))
        end = (center[0] + length * np.cos(angle),
               center[1] + length * np.sin(angle))
        return start, end

    def _new_trial(self, **kwargs):
        # Trial info. Default: 0th stimulus is also sample
        sample_angle = self.rng.uniform(0, 2*np.pi)
        sample_color = self.rng.uniform(0, 1, size=(3,))
        angles = [sample_angle]
        colors = [sample_color]
        for i in range(1, self.n_target):
            if self.rng.rand() > 0.5:
                new_angle = sample_angle + self.rng.choice([1, -1]) * self.delta_angle
                new_angle = np.mod(new_angle, 2*np.pi)
                angles.append(new_angle)
                colors.append(sample_color)
            else:
                new_color = sample_color + self.rng.permutation(self.delta_color)
                new_color = np.clip(new_color, 0, 1)
                angles.append(sample_angle)
                colors.append(new_color)

        centers = self.rng.permutation(self.target_centers)
        trial = {
            'angles': angles,
            'colors': colors,
            'centers': centers,
        }
        trial.update(kwargs)
        trial['ground_truth'] = centers[0]

        # Periods
        periods = ['fixation', 'sample', 'delay', 'decision']
        self.add_period(periods)

        # Observations
        fixation = (0, 0)
        for i in range(self.n_target):
            center = trial['centers'][i]
            angle = trial['angles'][i]
            color = tuple(trial['colors'][i])
            start, end = self._line_startend(center, angle, self.length)
            stim = visual.Line(self.win, lineWidth=self.lw,
                               lineColor=color,
                               start=start, end=end)
            self.add_ob(stim, 'decision')

            if i == 0:
                start, end = self._line_startend(fixation, angle, self.length)
                stim = visual.Line(self.win, lineWidth=self.lw,
                                   lineColor=color,
                                   start=start, end=end)
                self.add_ob(stim, 'sample')

        # Ground truth
        self.set_groundtruth(trial['ground_truth'], 'decision')

        return trial

    def _step(self, action):
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now
        # observations
        if self.in_period('fixation'):
            if np.sum(action**2) > 0.01:  # action = 0 means fixating
                new_trial = self.abort
                reward += self.rewards['abort']
        elif self.in_period('decision'):
            if np.sum(action**2) > 0.01:
                new_trial = True
                if np.sum((action - gt)**2) < 0.01:
                    reward += self.rewards['correct']
                    self.performance = 1
                else:
                    reward += self.rewards['fail']

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}
