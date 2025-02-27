from collections.abc import Callable
from pathlib import Path

import numpy as np
from gymnasium import Wrapper

import neurogym as ngym
from neurogym.config import Conf
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

    metadata: dict[str, str | None] = {  # noqa: RUF012
        "description": (
            "Saves relevant behavioral information: rewards, actions, observations, new trial, ground truth."
        ),
        "paper_link": None,
        "paper_name": None,
    }
    # TODO: use names similar to Tensorboard

    def __init__(
        self,
        env: ngym.TrialEnv,
        conf: Conf | str | Path | None = None,
        step_fn: Callable | None = None,
    ) -> None:
        super().__init__(env)
        self.env = env
        self.step_fn = step_fn
        if conf is None:
            conf = ngym.conf
        elif isinstance(conf, (str | Path)):
            conf = Conf(conf_file=conf)
        self.conf = conf
        self._configure_logger()

        self.data: dict[str, list] = {"action": [], "reward": []}
        if self.conf.monitor.trigger == "timestep":
            self.t = 0
        self.num_tr = 0

        # Paths
        if not self.conf.env.name:
            self.conf.env.name = self.env.__class__.__name__

        # Directory for saving plots
        save_dir_name = f"{self.conf.env.name}/{ngym.utils.iso_datetime()}"
        self.save_dir = ngym.utils.ensure_dir(self.conf.local_dir / save_dir_name)

        # Figures
        if self.conf.monitor.plot.save:
            self.stp_counter = 0
            self.ob_mat: list = []
            self.act_mat: list = []
            self.rew_mat: list = []
            self.gt_mat: list = []
            self.perf_mat: list = []

    def reset(self, seed=None):
        super().reset(seed=seed)
        return self.env.reset()

    def step(self, action):
        if self.step_fn:
            obs, rew, terminated, truncated, info = self.step_fn(action)
        else:
            obs, rew, terminated, truncated, info = self.env.step(action)
        if self.conf.monitor.plot.save:
            self.store_data(obs, action, rew, info)
        if self.conf.monitor.trigger == "timestep":
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
            save = (
                self.t >= self.conf.monitor.plot.interval
                if self.conf.monitor.trigger == "timestep"
                else self.num_tr % self.conf.monitor.plot.interval == 0
            )
            if save:
                np.savez(self.save_dir / f"trial-{self.num_tr!s}.npz", **self.data)
                if self.conf.log.verbose:
                    print("--------------------")
                    print("Number of steps: ", np.mean(self.num_tr))
                    print("Average reward: ", np.mean(self.data["reward"]))
                    print("--------------------")
                self.reset_data()
                if self.conf.monitor.plot.save:
                    self.stp_counter = 0
                if self.conf.monitor.trigger == "timestep":
                    self.t = 0
        return obs, rew, terminated, truncated, info

    def _configure_logger(self):
        ngym.logger.remove()
        ngym.logger.configure(**self.conf.log.make_config())
        ngym.logger.info("Logger configured.")

    def reset_data(self) -> None:
        for key in self.data:
            self.data[key] = []

    def store_data(self, obs, action, rew, info) -> None:
        if self.stp_counter <= self.conf.monitor.plot.interval:
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
            fname = self.save_dir / f"task_{self.num_tr:06d}.{self.conf.monitor.plot.ext}"
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
