#!/usr/bin/env python3

from pathlib import Path
from typing import ClassVar

import numpy as np
from gymnasium import Wrapper

from neurogym.utils.plotting import fig_


class Monitor(Wrapper):
    """Monitor task.

    Saves relevant behavioral information: rewards, actions, observations, new trial, ground truth.

    Args:
        folder: Folder where the data will be saved. (def: None, str)
            sv_per and sv_stp: Data will be saved every sv_per sv_stp's.
            (def: 100000, int)
        verbose: Whether to print information about average reward and number
            of trials. (def: False, bool)
        sv_fig: Whether to save a figure of the experiment structure. If True,
            a figure will be updated every sv_per. (def: False, bool)
        num_stps_sv_fig: Number of trial steps to include in the figure.
            (def: 100, int)
    """

    metadata: ClassVar[dict] = {
        "description": (
            "Saves relevant behavioral information: rewards, actions, observations, new trial, ground truth."
        ),
        "paper_link": None,
        "paper_name": None,
    }
    # TODO: use names similar to Tensorboard

    def __init__(
        self,
        env,
        folder=None,
        sv_per=100000,
        sv_stp="trial",
        verbose=False,
        sv_fig=False,
        num_stps_sv_fig=100,
        name="",
        fig_type="png",
        step_fn=None,
    ) -> None:
        super().__init__(env)
        self.env = env
        self.num_tr = 0
        self.step_fn = step_fn
        # data to save
        self.data = {"action": [], "reward": []}
        self.sv_per = sv_per
        self.sv_stp = sv_stp
        self.fig_type = fig_type
        if self.sv_stp == "timestep":
            self.t = 0
        self.verbose = verbose
        if folder is None:
            # FIXME is it ok to use tempfile.TemporaryDirectory instead or does this need to be stored locally always?
            self.folder = "tmp"
        Path(self.folder).mkdir(parents=True, exist_ok=True)
        # seeding
        self.sv_name = self.folder + self.env.__class__.__name__ + "_bhvr_data_" + name + "_"  # FIXME: use pathlib
        # figure
        self.sv_fig = sv_fig
        if self.sv_fig:
            self.num_stps_sv_fig = num_stps_sv_fig
            self.stp_counter = 0
            self.ob_mat = []
            self.act_mat = []
            self.rew_mat = []
            self.gt_mat = []
            self.perf_mat = []

    def reset(self, seed=None):
        super().reset(seed=seed)
        return self.env.reset()

    def step(self, action):
        if self.step_fn:
            obs, rew, terminated, truncated, info = self.step_fn(action)
        else:
            obs, rew, terminated, truncated, info = self.env.step(action)
        if self.sv_fig:
            self.store_data(obs, action, rew, info)
        if self.sv_stp == "timestep":
            self.t += 1
        if info["new_trial"]:
            self.num_tr += 1
            self.data["action"].append(action)
            self.data["reward"].append(rew)
            for key in info:
                if key not in self.data:
                    self.data[key] = [info[key]]
                else:
                    self.data[key].append(info[key])

            # save data
            save = False
            save = self.t >= self.sv_per if self.sv_stp == "timestep" else self.num_tr % self.sv_per == 0
            if save:
                np.savez(self.sv_name + str(self.num_tr) + ".npz", **self.data)  # FIXME: use pathlib
                if self.verbose:
                    print("--------------------")
                    print("Number of steps: ", np.mean(self.num_tr))
                    print("Average reward: ", np.mean(self.data["reward"]))
                    print("--------------------")
                self.reset_data()
                if self.sv_fig:
                    self.stp_counter = 0
                if self.sv_stp == "timestep":
                    self.t = 0
        return obs, rew, terminated, truncated, info

    def reset_data(self) -> None:
        for key in self.data:
            self.data[key] = []

    def store_data(self, obs, action, rew, info) -> None:
        if self.stp_counter <= self.num_stps_sv_fig:
            self.ob_mat.append(obs)
            self.act_mat.append(action)
            self.rew_mat.append(rew)
            if "gt" in info:
                self.gt_mat.append(info["gt"])
            else:
                self.gt_mat.append(-1)
            if "performance" in info:
                self.perf_mat.append(info["performance"])
            else:
                self.perf_mat.append(-1)
            self.stp_counter += 1
        elif len(self.rew_mat) > 0:
            fname = self.sv_name + f"task_{self.num_tr:06d}.{self.fig_type}"
            obs_mat = np.array(self.ob_mat)
            act_mat = np.array(self.act_mat)
            fig_(
                ob=obs_mat,
                actions=act_mat,
                gt=self.gt_mat,
                rewards=self.rew_mat,
                performance=self.perf_mat,
                fname=fname,
            )
            self.ob_mat = []
            self.act_mat = []
            self.rew_mat = []
            self.gt_mat = []
            self.perf_mat = []
