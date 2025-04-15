from collections.abc import Callable
from pathlib import Path

import numpy as np
from gymnasium import Wrapper

import neurogym as ngym
from neurogym.utils.plotting import fig_


class Monitor(Wrapper):
    """Monitor task.

    Saves relevant behavioral information: rewards, actions, observations, new trial, ground truth.

    Args:
        env: The wrapped environment.
        config: An optional configuration (either a `Config` object or a `str` or `Path` object
            pointing to a TOML configuration file). Defaults to None.
        step_function: An optional step function to override the built-in `step()`
            method provided by the environment. Defaults to None.
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
        config: ngym.Config | str | Path | None = None,
        step_function: Callable | None = None,
    ) -> None:
        super().__init__(env)
        self.env = env
        self.step_function = step_function
        if config is None:
            config = ngym.config
        elif isinstance(config, (str, Path)):
            config = ngym.Config(config_file=config)
        self.config: ngym.Config = config
        self._configure_logger()

        self.data: dict[str, list] = {"action": [], "reward": []}
        if self.config.monitor.plot.trigger == "step":
            self.t = 0
        self.num_tr = 0

        # Paths
        if not self.config.env.name:
            self.config.env.name = self.env.__class__.__name__

        # Directory for saving plots
        save_dir_name = f"{self.config.env.name}/{ngym.utils.iso_timestamp()}"
        self.save_dir = ngym.utils.ensure_dir(self.config.local_dir / save_dir_name)

        # Figures
        if self.config.monitor.plot.create:
            self.stp_counter = 0
            self.ob_mat: list = []
            self.act_mat: list = []
            self.rew_mat: list = []
            self.gt_mat: list = []
            self.perf_mat: list = []

    def _configure_logger(self):
        ngym.logger.remove()
        ngym.logger.configure(**self.config.monitor.log.make_config())
        ngym.logger.info("Logger configured.")

    def reset(self, seed=None):
        super().reset(seed=seed)
        return self.env.reset()

    def step(self, action):
        if self.step_function:
            obs, rew, terminated, truncated, info = self.step_function(action)
        else:
            obs, rew, terminated, truncated, info = self.env.step(action)
        if self.config.monitor.plot.create:
            self.store_data(obs, action, rew, info)
        if self.config.monitor.plot.trigger == "step":
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
            save = (
                self.t >= self.config.monitor.plot.interval
                if self.config.monitor.plot.trigger == "step"
                else self.num_tr % self.config.monitor.plot.interval == 0
            )
            if save:
                np.savez(self.save_dir / f"trial-{self.num_tr!s}.npz", **self.data)
                if self.config.monitor.log.verbose:
                    print("--------------------")
                    print("Number of trials: ", self.num_tr)
                    print("Average reward: ", np.mean(self.data["reward"]))
                    print("--------------------")
                self.reset_data()
                if self.config.monitor.plot.create:
                    self.stp_counter = 0
                if self.config.monitor.plot.trigger == "step":
                    self.t = 0
        return obs, rew, terminated, truncated, info

    def reset_data(self) -> None:
        for key in self.data:
            self.data[key] = []

    def store_data(self, obs, action, rew, info) -> None:
        if self.stp_counter <= self.config.monitor.plot.interval:
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
            fname = self.save_dir / f"task_{self.num_tr:06d}.{self.config.monitor.plot.ext}"
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
