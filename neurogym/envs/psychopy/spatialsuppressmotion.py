# This is try to code spatial suppression motion task


import sys

import numpy as np
from gymnasium import spaces
from numpy.polynomial.polynomial import polyfit
from psychopy import visual
from scipy.interpolate import interp1d

from .psychopy_env import PsychopyEnv


class SpatialSuppressMotion(PsychopyEnv):
    """Spatial suppression motion task.

    This task is useful to study center-surround interaction in monkey MT and human
    psychophysical performance in motion perception. By Ru-Yuan Zhang (ruyuanzhang@gmail.com).

    Tha task is derived from (Tadin et al. Nature, 2003). In this task, there is no fixation or decision stage. We only
    present a stimulus and a subject needs to perform a 4-AFC motion direction judgement. The ground-truth is the
    probabilities for choosing the four directions at a given time point. The probabilities depend on stimulus contrast
    and size and the probabilities are derived from emprically measured human psychophysical performance.

    Args:
        <dt>: millisecs per image frame, default: 8.3 (given 120HZ monitor)
        <win_size>: size per image frame
        <timing>: millisecs, stimulus duration, default: 8.3 * 36 frames ~ 300 ms.
            This is the longest duration we need (i.e., probability reach ceilling)

    Note that please input default seq_len = 36 frames when creating dataset object.


    """

    metadata = {  # noqa: RUF012
        "paper_link": "https://www.nature.com/articles/nature01800",
        "paper_name": """Perceptual consequences of centre-surround antagonism in visual motion processing """,
        "tags": ["perceptual", "plaid", "motion", "center-surround"],
    }

    def __init__(
        self,
        dt=8.3,
        win_kwargs=None,
        timing=None,
        rewards=None,
    ) -> None:
        if timing is None:
            timing = {"stimulus": 300}
        if win_kwargs is None:
            win_kwargs = {"size": (100, 100)}
        super().__init__(dt=dt, win_kwargs=win_kwargs)

        # Rewards
        self.rewards = {"abort": -0.1, "correct": +1.0, "fail": 0.0}
        if rewards:
            self.rewards.update(rewards)

        # Timing
        self.timing = {
            "stimulus": 300,  # we only need stimulus period for psychophysical task
        }
        if timing:
            self.timing.update(timing)

        self.win.color = 0  # set it to gray background, -1, black;1,white

        self.abort = False

        # four directions
        self.action_space = spaces.Box(
            0,
            1,
            shape=(4,),
            dtype=np.float32,
        )  # the probabilities for four direction

        self.directions = [1, 2, 3, 4]  # motion direction left/right/up/down
        self.directions_component = [
            (-1, 1),
            (1, -1),
            (-1, -1),
            (1, 1),
        ]  # direction of two grating component to control plaid direction left/right/up/down
        self.directions_anti = [2, 1, 4, 3]
        self.directions_ortho = [[3, 4], [3, 4], [1, 2], [1, 2]]

    def _new_trial(self, diameter=None, contrast=None, direction=None):
        """Define a new stimulus.

        Args:
            diameter: 0~1, stimulus size in norm units
            contrast: 0~1, stimulus contrast
            direction: int(1/2/3/4), left/right/up/down.
        """
        # if no stimulus information provided, we random sample stimulus parameters
        if direction is None:
            direction = self.rng.choice(self.directions)
        if diameter is None:
            diameter = 1  # we fixed for now
        if contrast is None:
            contrast = self.rng.choice([0.05, 0.99])  # Low contrast, and high contrast

        trial = {
            "diameter": diameter,
            "contrast": contrast,
            "direction": direction,
        }

        # define some motion parameters.
        # We assume 16 deg for the full FOV (8 deg radius)
        degrees = 16
        cycles_per_degree = 0.8  # cycles / degree
        speed = 4  # deg / sec
        cycles_per_fov = cycles_per_degree * degrees  # cycles per fov
        temporal_frequency = cycles_per_degree * speed  # temporal frequency cycles / sec
        comp_direct = self.directions_component[trial["direction"] - 1]

        # obtain the temporal contrast profile and mv_length
        profile, _ = self.envelope(0.1)  # 0.1 sec

        # Periods and Timing
        # we only need stimulus period for this psychophysical task
        periods = ["stimulus"]
        self.add_period(periods)

        # We need ground_truth
        # the probablities to choose four directions given stimulus parameters
        trial["ground_truth"] = self.getgroundtruth(trial)

        # Observation
        if sys.platform == "darwin":  # trick for darwin mac window
            diameter = trial["diameter"] * 2

        grating1 = visual.GratingStim(
            self.win,
            mask="raisedCos",
            opacity=1.0,
            size=diameter,
            sf=(cycles_per_fov, 0),
            ori=45,
            contrast=trial["contrast"],
        )
        grating2 = visual.GratingStim(
            self.win,
            mask="raisedCos",
            opacity=0.5,
            size=diameter,
            sf=(cycles_per_fov, 0),
            ori=135,
            contrast=trial["contrast"],
        )

        # create the movie
        ob = self.view_ob(period="stimulus")
        for i in range(ob.shape[0]):
            grating1.contrast = trial["contrast"] * profile[i]
            # drift it, comp_direct control the direction
            grating1.phase = comp_direct[0] * temporal_frequency * i * self.dt / 1000
            grating1.draw()

            grating2.contrast = trial["contrast"] * profile[i]
            grating2.phase = comp_direct[1] * temporal_frequency * i * self.dt / 1000
            grating2.draw()

            self.win.flip()
            im = self.win._getFrame()  # noqa: SLF001
            im = np.array(im)  # convert it to numpy array, it is a nPix x nPix x 3 array

            # Here we did not use .add_ob function of psychopyEnv object
            ob[i] = im.copy()  # we switch the add, which seems wrong for image

        # Ground truth
        self.set_groundtruth(trial["ground_truth"], "stimulus")

        return trial

    def _step(self, action):  # noqa: ARG002
        # We only need output at very end, no need to check action every step and calculate reward.
        # Just let this function complete all steps.
        # The _step function is useful for making a choice early in a trial or the situation when breaking the fixation.

        new_trial = False
        terminated = False
        truncated = False
        # rewards
        reward = 0
        gt = self.gt_now
        return (
            self.ob_now,
            reward,
            terminated,
            truncated,
            {"new_trial": new_trial, "gt": gt},
        )

    @staticmethod
    def envelope(time_sigma, frame_rate=120, cut_off=True, amplitude=128):
        """Create a temporal profile and mv_length to motion stimulus in the spatial suppression task.

        This function is modified fron Duje Tadin's code.

        Not critical, we can use a square temporal profile

        Arg:
            <time_sigma>: in secs, sigma of temporal envelope
            <frame_rate>: int, hz, monitor frame rate
            <cut_off>: default: True. Whether to cut
            <amplitude>: int, 128.

        We return the tempora modulation <profile> and an int indicating <mv_length>.

        This function is adoped from Duje Tadin. Some variables here are not clear.
        """
        time_sigma *= 1000  # convert it to millisecs
        gauss_only = 0 if cut_off else 1

        fr = round(frame_rate / 20)  # this frame is determined arbitrarily
        xx = np.arange(fr) + 1

        k = 0
        tt = np.arange(7, 25 + 0.25, 0.25)
        x1, cum1 = np.empty(tt.size), np.empty(tt.size)
        for k, time1 in enumerate(tt):
            x1[k] = time1
            time = time1 / (1000 / frame_rate)
            time_gauss = np.exp(-(((xx) / (np.sqrt(2) * time)) ** 2))
            cum1[k] = np.sum(time_gauss) * 2

        # we obtain a relation between underlying area and time
        p, _ = polyfit(cum1, x1, deg=2, full=True)
        area = time_sigma * frame_rate / 400
        if cut_off > -1:
            uniform = int(np.floor(area - 3))
            if time_sigma > cut_off & ~gauss_only:  # we provide Gaussian edges and a plateao part
                remd = area - uniform
                time = p[2] * remd**2 + p[1] * remd + p[0]
                time /= 1000 / frame_rate  # how many frame

                # calculate the gaussian part
                del xx
                xx = np.arange(fr) + 1
                time_gauss = np.exp(-(((xx) / (np.sqrt(2) * time)) ** 2))

                # add time_gauss to both sides of the temporal profile
                profile = np.ones(uniform + 2 * fr)
                profile[:fr] = time_gauss[::-1]
                profile[-time_gauss.size :] = time_gauss

            else:  # in this case, we provide a completely Gaussian profile, with no flat part
                time = time_sigma / (1000 / frame_rate)
                mv_length = time * (1000 / frame_rate) * 6
                mv_length = round((round(mv_length / (1000 / frame_rate))) / 2) * 2 + 1
                xx = np.arange(mv_length) + 1
                xx -= xx.mean()
                profile = np.exp(-(((xx) / (np.sqrt(2) * time)) ** 2))

            # we trim the frame that are very small
            small = (amplitude * profile < 0.5).sum() / 2
            profile = profile[int(small) : profile.size - int(small)]
            mv_length = profile.size

        else:  # in this case, only give a flat envelope
            mv_length = round(area)
            profile = np.ones(mv_length)

        return profile, mv_length

    def getgroundtruth(self, trial):
        """The utility function to obtain ground truth probabilities for four direction.

        Input trial is a dict, contains fields <duration>, <contrast>, <diameter>, and <direction>.

        We output a (4,) tuple indicate the probabilities to perceive left/right/up/down direction.
        This label comes from emprically measured human performance.
        """
        frame_ind = [8, 9, 10, 13, 15, 18, 21, 28, 36, 37, 38, 39]
        xx = [1, 2, 3, 4, 5, 6, 7]
        yy = [0.249] * 7

        frame_ind = xx + frame_ind  # to fill in the first a few frames
        frame_ind = [i - 1 for i in frame_ind]  # frame index start from

        seq_len = self.view_ob(period="stimulus").shape[0]
        xnew = np.arange(seq_len)

        if trial["contrast"] == 0.99:
            # large size (11 deg radius), High contrast
            prob_corr = [*yy, 0.249, 0.249, 0.249, 0.27, 0.32, 0.4583, 0.65, 0.85, 0.99, 0.99, 0.99, 0.99]
            prob_anti = [*yy, 0.249, 0.29, 0.31, 0.4, 0.475, 0.4167, 0.3083, 0.075, 0.04, 0.04, 0.03, 0.03]

        elif trial["contrast"] == 0.05:
            # large size (11 deg radius), low contrast
            prob_corr = [*yy, 0.25, 0.26, 0.2583, 0.325, 0.45, 0.575, 0.875, 0.933, 0.99, 0.99, 0.99, 0.99]
            prob_anti = [*yy, 0.25, 0.26, 0.2583, 0.267, 0.1417, 0.1167, 0.058, 0.016, 0.003, 0.003, 0.003, 0.003]

        corr_prob = interp1d(
            frame_ind,
            prob_corr,
            kind="slinear",
            fill_value="extrapolate",
        )(xnew)
        anti_prob = interp1d(
            frame_ind,
            prob_anti,
            kind="slinear",
            fill_value="extrapolate",
        )(xnew)
        ortho_prob = (1 - (corr_prob + anti_prob)) / 2

        direction = trial["direction"] - 1
        direction_anti = self.directions_anti[direction] - 1
        direction_ortho = [i - 1 for i in self.directions_ortho[direction]]

        gt = np.zeros((4, seq_len))
        gt[direction, :] = corr_prob
        gt[direction_anti, :] = anti_prob
        gt[direction_ortho, :] = ortho_prob

        return gt.T
        # gt is a seq_len x 4 numpy array
