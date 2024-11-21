import sys
from typing import Any

import numpy as np

try:
    from psychopy import visual
except ImportError as e:
    msg = "Psychopy is not installed."
    raise ImportError(msg) from e

from gymnasium import spaces

import neurogym as ngym


class PsychopyEnv(ngym.TrialEnv):
    """Superclass for environments with psychopy stimuli."""

    def __init__(self, win_kwargs=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # fix the bug when multi windows with different sizes in a batch
        win_kwargs_tmp: dict[str, Any] = (
            {"size": (100, 100), "color": "black"} if win_kwargs is None else win_kwargs.copy()
        )

        if sys.platform == "darwin":
            # TODO: Check if this works across platform
            win_kwargs_tmp["size"] = tuple(int(dim) // 2 for dim in win_kwargs_tmp["size"])
        # psychopy window kwargs can be supplied by 'win_kws'
        self.win = visual.Window(**win_kwargs_tmp)
        self.win.backend.winHandle.set_visible(False)
        self.win.flip()
        im = self.win._getFrame()  # noqa: SLF001
        value = np.array(im)
        self._default_ob_value = value[0, 0]  # corner pixel, array 3-channels

        ob_shape = (self.win.size[0], self.win.size[1], 3)
        self.observation_space = spaces.Box(0, 255, shape=ob_shape, dtype=np.uint8)

    def add_ob(self, value, period=None, where=None) -> None:
        if isinstance(value, visual.BaseVisualStim):
            if where is not None:
                print(
                    "Warning: Setting where to values other than Nonehas no effect when adding PsychoPy stimuli",
                )

            if isinstance(value, visual.DotStim):
                # TODO: Check other types of stimuli as well
                # These stimuli need to be drawn every frame
                if not (isinstance(period, str) or period is None):
                    msg = f"{period=} not supported for stimulus {value}."
                    raise ValueError(msg)

                ob = self.view_ob(period=period)
                for i in range(ob.shape[0]):
                    value.draw()
                    self.win.flip()
                    im = self.win._getFrame()  # noqa: SLF001
                    ob[i] += np.array(im)
            else:
                # Static stimuli
                value.draw()
                self.win.flip()
                im = self.win._getFrame()  # noqa: SLF001
                super().add_ob(np.array(im), period, where)
        else:
            super().add_ob(value, period, where)
