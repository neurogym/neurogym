import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces

import neurogym as ngym


class CVLearning(ngym.TrialEnv):
    """Implements shaping for the delay-response task.

    Agents have to integrate two stimuli and report which one is larger on average after a delay.

    Args:
        stim_scale: Controls the difficulty of the experiment. (def: 1., float)
        max_num_reps: Maximum number of times that agent can go in a row
        to the same side during phase 0. (def: 3, int)
        th_stage: Performance threshold needed to proceed to the following
        phase. (def: 0.7, float)
        keep_days: Number of days that the agent will be kept in the same phase
        once arrived to the goal performacance. (def: 1, int)
        trials_day: Number of trials performed during one day. (def: 200, int)
        perf_len: Number of trials used to compute instantaneous performance.
        (def: 20, int)
        stages: Stages used to train the agent. (def: [0, 1, 2, 3, 4], list)
    """

    metadata = {  # noqa: RUF012
        "paper_link": "https://www.nature.com/articles/s41586-019-0919-7",
        "paper_name": "Discrete attractor dynamics underlies persistent activity in the frontal cortex",
        "tags": ["perceptual", "delayed response", "two-alternative", "supervised"],
    }

    def __init__(
        self,
        dt=100,
        rewards=None,
        timing=None,
        stim_scale=1.0,
        sigma=1.0,
        max_num_reps=3,
        th_stage=0.7,
        keep_days=1,
        trials_day=300,
        perf_len=20,
        stages=None,
        n_ch=10,
    ) -> None:
        if stages is None:
            stages = [0, 1, 2, 3, 4]
        super().__init__(dt=dt)
        self.choices = [1, 2]
        self.n_ch = n_ch  # number of obs and actions different from fixation
        # cohs specifies the amount of evidence
        # (which is modulated by stim_scale)
        self.cohs = np.array([0, 6.4, 12.8, 25.6, 51.2]) * stim_scale
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise

        # Rewards
        self.rewards = {"abort": -0.1, "correct": +1.0, "fail": -1.0}
        if rewards:
            self.rewards.update(rewards)

        self.delay_durs = [1000, 3000]
        self.timing = {
            "fixation": 200,
            "stimulus": 1150,
            "delay": lambda: self.rng.uniform(*self.delay_durs),
            "decision": 1500,
        }
        if timing:
            self.timing.update(timing)

        self.stages = stages

        self.r_fail = self.rewards["fail"]
        self.action = 0
        self.abort = False
        self.firstcounts = True  # whether trial ends at first attempt
        self.first_flag = False  # whether first attempt has been done
        self.ind = 0  # index of the current stage
        if th_stage == -1:
            self.curr_ph = self.stages[-1]
        else:
            self.curr_ph = self.stages[self.ind]
        self.rew = 0

        # PERFORMANCE VARIABLES
        self.trials_counter = 0
        # Day/session performance
        self.curr_perf: np.floating = np.float64(0)
        self.trials_day: int = trials_day
        self.th_perf = th_stage
        self.day_perf: np.ndarray = np.empty(trials_day)
        self.w_keep = [keep_days] * len(self.stages)  # TODO: simplify??
        # number of days to keep an agent on a stage
        # once it has reached th_perf
        self.days_keep = self.w_keep[self.ind]
        self.keep_stage = False  # wether the agent can move to the next stage
        # Instantaneous performance (moving window)
        self.inst_perf: np.floating = np.float64(0)
        self.perf_len = perf_len  # window length
        self.mov_perf: np.ndarray = np.zeros(perf_len)

        # STAGE VARIABLES
        # stage 0
        # max number of consecutive times that an agent can repeat an action
        # receiving positive reward on stage 0
        self.max_num_reps = max_num_reps
        # counter of consecutive actions at the same side
        self.action_counter = 0
        # stage 2
        # min performance to keep the agent in stage 2
        self.min_perf = 0.5  # TODO: no magic numbers
        self.stage_reminder = False  # control if a stage has been explored
        # stage 3
        self.inc_delays = 0  # proportion of the total delays dur to keep
        self.delay_milestone = 0  # delays durs at the beggining of a day
        # proportion of the total delays dur to incease every time that the
        # agent reaches a threshold performance
        self.inc_factor = 0.25
        self.inc_delays_th = th_stage  # th perf to increase delays in stage 3
        self.dec_delays_th = 0.5  # th perf to decrease delays in stage 3
        # number of trials spent on a specific delays duration
        self.trials_delay = 0
        self.max_delays = True  # wheter delays have reached their max dur
        self.dur = [0] * len(self.delay_durs)

        # action and observation spaces
        self.action_space = spaces.Discrete(n_ch + 1)
        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            shape=(n_ch + 1,),
            dtype=np.float32,
        )

    def _new_trial(self, **kwargs):
        """Called when a trial ends to generate the next trial.

        The following variables are created:
            durations: Stores the duration of the different periods.
            ground truth: Correct response for the trial.
            coh: Stimulus coherence (evidence) for the trial.
            obs: Observation.
        """
        self.set_phase()
        if self.curr_ph == 0:
            # control that agent does not repeat side more than 3 times
            self.count(self.action)

        trial = {
            "ground_truth": self.rng.choice(self.choices),
            "coh": self.rng.choice(self.cohs),
            "sigma": self.sigma,
        }

        # init durations with None
        self.durs = dict.fromkeys(self.timing)
        self.firstcounts = True

        self.first_choice_rew = None
        if self.curr_ph == 0:
            # no stim, reward is in both left and right
            # agent cannot go N times in a row to the same side
            if np.abs(self.action_counter) >= self.max_num_reps:
                ground_truth = 1 if self.action == 2 else 2
                trial.update({"ground_truth": ground_truth})
                self.rewards["fail"] = 0
            else:
                self.rewards["fail"] = self.rewards["correct"]
            self.durs.update({"stimulus": 0, "delay": 0})
            trial.update({"sigma": 0})

        elif self.curr_ph == 1:
            # stim introduced with no ambiguity
            # wrong answer is not penalized
            # agent can keep exploring until finding the right answer
            self.durs.update({"delay": 0})
            trial.update({"coh": 100})
            trial.update({"sigma": 0})
            self.rewards["fail"] = 0
            self.firstcounts = False
        elif self.curr_ph == 2:
            # first answer counts
            # wrong answer is penalized
            self.durs.update({"delay": (0)})
            trial.update({"coh": 100})
            trial.update({"sigma": 0})
            self.rewards["fail"] = self.r_fail
        elif self.curr_ph == 3:
            self.rewards["fail"] = self.r_fail
            # increasing or decreasing delays durs
            if self.trials_delay > self.perf_len:
                if self.inst_perf >= self.inc_delays_th and self.inc_delays < 1:
                    self.inc_delays += self.inc_factor
                    self.trials_delay = 0
                elif self.inst_perf <= self.dec_delays_th and self.inc_delays > self.delay_milestone:
                    self.inc_delays -= self.inc_factor
                    self.trials_delay = 0
            self.dur = [int(d * self.inc_delays) for d in self.delay_durs]
            if self.dur == self.delay_durs:
                self.max_delays = True
            else:
                self.max_delays = False
            rng = np.random.default_rng()
            self.durs.update({"delay": rng.choice(self.dur)})
            # delay component is introduced
            trial.update({"coh": 100})
            trial.update({"sigma": 0})
        # phase 4: ambiguity component is introduced

        self.first_flag = False

        # ---------------------------------------------------------------------
        # Trial
        # ---------------------------------------------------------------------

        trial.update(kwargs)

        # ---------------------------------------------------------------------
        # Periods
        # ---------------------------------------------------------------------
        self.add_period("fixation")
        self.add_period("stimulus", duration=self.durs["stimulus"], after="fixation")
        self.add_period("delay", duration=self.durs["delay"], after="stimulus")
        self.add_period("decision", after="delay")

        # define observations
        self.set_ob([1] + [0] * self.n_ch, "fixation")
        stim = self.view_ob("stimulus")
        stim[:, 0] = 1
        stim[:, 1:3] = (1 - trial["coh"] / 100) / 2
        stim[:, trial["ground_truth"]] = (1 + trial["coh"] / 100) / 2
        stim[:, 3:] = 0.5
        stim[:, 1:] += self.rng.randn(stim.shape[0], self.n_ch) * trial["sigma"]
        self.set_ob([1] + [0] * self.n_ch, "delay")

        self.set_groundtruth(trial["ground_truth"], "decision")

        return trial

    def count(self, action) -> None:
        """Check the last three answers during stage 0 so the network has to alternate between left and right."""
        if action != 0:
            new = action - 2 / action
            if np.sign(self.action_counter) == np.sign(new):
                self.action_counter += new
            else:
                self.action_counter = new

    def set_phase(self) -> None:
        self.day_perf[self.trials_counter] = 1 * (self.rew == self.rewards["correct"])
        self.mov_perf[self.trials_counter % self.perf_len] = 1 * (self.rew == self.rewards["correct"])
        self.trials_counter += 1
        self.trials_delay += 1

        # Instantaneous perfromace
        if self.trials_counter > self.perf_len:
            self.inst_perf = np.mean(self.mov_perf)
            if self.inst_perf < self.min_perf and self.curr_ph == 2:
                if 1 in self.stages:
                    self.curr_ph = 1
                    self.stage_reminder = True
                    self.ind -= 1
            elif self.inst_perf > self.th_perf and self.stage_reminder:
                self.curr_ph = 2
                self.ind += 1
                self.stage_reminder = False

        # End of the day
        if self.trials_counter >= self.trials_day:
            self.trials_counter = 0
            self.curr_perf = np.mean(self.day_perf)
            self.day_perf = np.empty(self.trials_day)
            self.delay_milestone = self.inc_delays
            # Keeping or changing stage
            if self.curr_perf >= self.th_perf and self.max_delays:
                self.keep_stage = True
            else:
                self.keep_stage = False
                self.days_keep = self.w_keep[self.ind]
            if self.keep_stage:
                if self.days_keep <= 0 and self.curr_ph < self.stages[-1]:
                    self.ind += 1
                    self.curr_ph = self.stages[self.ind]
                    self.days_keep = self.w_keep[self.ind] + 1
                    self.keep_stage = False
                self.days_keep -= 1

    def _step(self, action):
        new_trial = False
        terminated = False
        truncated = False
        # rewards
        reward = 0
        gt = self.gt_now
        first_choice = False
        if action != 0 and not self.in_period("decision"):
            new_trial = self.abort
            reward = self.rewards["abort"]
        elif self.in_period("decision"):
            if action == gt:
                reward = self.rewards["correct"]
                new_trial = True
                if not self.first_flag:
                    first_choice = True
                    self.first_flag = True
                    self.performance = 1
            elif action == 3 - gt:  # 3-action is the other act
                reward = self.rewards["fail"]
                new_trial = self.firstcounts
                if not self.first_flag:
                    first_choice = True
                    self.first_flag = True
                    self.performance = self.rewards["fail"] == self.rewards["correct"]

        # check if first choice (phase 1)
        if not self.firstcounts and first_choice:
            self.first_choice_rew = reward
        # set reward for all phases
        self.rew = self.first_choice_rew or reward

        if new_trial and self.curr_ph == 0:
            self.action = action

        info = {
            "new_trial": new_trial,
            "gt": gt,
            "num_tr": self.num_tr,
            "curr_ph": self.curr_ph,
            "first_rew": self.rew,
            "keep_stage": self.keep_stage,
            "inst_perf": self.inst_perf,
            "trials_day": self.trials_counter,
            "durs": self.dur,
            "inc_delays": self.inc_delays,
            "curr_perf": self.curr_perf,
            "trials_count": self.trials_counter,
            "th_perf": self.th_perf,
            "num_stps": self.t_ind,
        }
        return self.ob_now, reward, terminated, truncated, info


if __name__ == "__main__":
    plt.close("all")
    env = CVLearning(stages=[0, 2, 3, 4], trials_day=2, keep_days=1)
    data = ngym.utils.plot_env(env, num_steps=200)
    env = CVLearning(stages=[3, 4], trials_day=2, keep_days=1)
    data = ngym.utils.plot_env(env, num_steps=200)
