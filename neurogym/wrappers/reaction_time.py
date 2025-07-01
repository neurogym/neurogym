"""Noise wrapper.

Created on Thu Feb 28 15:07:21 2019

@author: molano
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym

if TYPE_CHECKING:
    from neurogym.core import TrialEnv


class ReactionTime(gym.Wrapper):  # TODO: Make this a trial wrapper instead?
    """Allow reaction time response.

    Modifies a given environment by allowing the network to act at
    any time after the fixation period. By default, the trial ends when the
    stimulus period ends. Optionally, the original trial structure can be
    preserved while still allowing early responses.

    Args:
        env: The environment to wrap
        urgency: Urgency signal added to reward at each timestep
        end_on_stimulus: If True (default), trial ends when
            stimulus ends. If False, preserves original trial timing while
            allowing early responses during stimulus period.
    """

    metadata: dict[str, str | None] = {  # noqa: RUF012
        "description": (
            "Modifies a given environment by allowing the network to act at any time after the fixation period. "
            "The trial ends when the stimulus period ends by default, or preserves original timing if specified."
        ),
        "paper_link": None,
        "paper_name": None,
    }

    def __init__(self, env: TrialEnv, urgency: float = 0.0, end_on_stimulus: bool = True) -> None:
        super().__init__(env)
        self.urgency = urgency
        self.end_on_stimulus = end_on_stimulus
        self.tr_dur = 0

    def reset(self, seed=None, options=None):
        step_fn = options.get("step_fn") if options else None
        if step_fn is None:
            step_fn = self.step
        return self.env.reset(options={"step_fn": step_fn}, seed=seed)

    def step(self, action):
        dec = "decision"
        stim = "stimulus"
        if self.env.t_ind == 0:
            try:
                # Dictionary content access - works without `unwrapped` because
                # `__getattr__` retrieves the shared dictionary object from base environment
                original_gt = self.env.gt[self.env.start_ind[dec]]
                # Dictionary key assignment - works without `unwrapped` because we're modifying
                # the contents of the shared dictionary object, not reassigning the attribute
                self.env.start_t[dec] = self.env.start_t[stim] + self.env.dt

                if self.end_on_stimulus:
                    # Use `unwrapped` to modify base environment's `tmax` directly.
                    # `self.env.tmax = value` would create new attribute on wrapper,
                    # but base environment's `step()` checks its own `self.tmax`.
                    self.env.unwrapped.tmax = self.env.end_t[stim]

                self.env.gt[self.env.start_ind[stim] + 1 : self.env.end_ind[stim]] = original_gt
            except AttributeError as e:
                msg = "ReactionTime wrapper requires 'stimulus' and 'decision' periods."
                raise AttributeError(msg) from e

        obs, reward, terminated, truncated, info = self.env.step(action)

        if info.get("new_trial", False):
            self.tr_dur = 0
            info["tr_dur"] = self.tr_dur
        else:
            self.tr_dur = self.env.t_ind
        reward += self.urgency
        return obs, reward, terminated, truncated, info
