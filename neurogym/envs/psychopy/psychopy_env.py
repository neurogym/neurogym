import sys

import numpy as np
try:
    from psychopy import visual
except ImportError as e:
    raise ImportError('Psychopy is not installed.')

from gym import spaces
import neurogym as ngym


class PsychopyEnv(ngym.TrialEnv):
    """Superclass for environments with psychopy stimuli."""

    def __init__(self, win_kwargs=None, *args, **kwargs):
        super(PsychopyEnv, self).__init__(*args, **kwargs)

        if win_kwargs is None:
            win_kwargs_tmp={'size': (100, 100), 'color': 'black'}
        else: 
            win_kwargs_tmp = win_kwargs.copy() # fix the bug when multi windows with different sizes in a batch  
        
        if sys.platform == 'darwin':
            # TODO: Check if this works across platform
            win_kwargs_tmp['size'] = (int(win_kwargs_tmp['size'][0]/2),
                                      int(win_kwargs_tmp['size'][1]/2))
        # psychopy window kwargs can be supplied by 'win_kws'
        self.win = visual.Window(**win_kwargs_tmp)
        self.win.backend.winHandle.set_visible(False)
        self.win.flip()
        im = self.win._getFrame()
        value = np.array(im)
        self._default_ob_value = value[0, 0]  # corner pixel, array 3-channels

        ob_shape = (self.win.size[0], self.win.size[1], 3)
        self.observation_space = spaces.Box(0, 255, shape=ob_shape,
                                            dtype=np.uint8)

    def add_ob(self, value, period=None, where=None):
        if isinstance(value, visual.BaseVisualStim):
            if where is not None:
                print('Warning: Setting where to values other than None'
                      'has no effect when adding PsychoPy stimuli')

            if isinstance(value, visual.DotStim):
                # TODO: Check other types of stimuli as well
                # These stimuli need to be drawn every frame
                if not (isinstance(period, str) or period is None):
                    raise ValueError('Period {:s} not '.format(str(period)) +
                                     'supported for stimuli {:s}'.format(str(value)))

                ob = self.view_ob(period=period)
                for i in range(ob.shape[0]):
                    value.draw()
                    self.win.flip()
                    im = self.win._getFrame()
                    ob[i] += np.array(im)
            else:
                # Static stimuli
                value.draw()
                self.win.flip()
                im = self.win._getFrame()
                super().add_ob(np.array(im), period, where)
        else:
            super().add_ob(value, period, where)
