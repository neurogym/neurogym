"""Test wrappers."""

import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pytest

import neurogym as ngym
from neurogym.wrappers import (
    Noise,
    PassAction,
    PassReward,
    ReactionTime,
    SideBias,
)

# ruff: noqa: N803, F821


@pytest.mark.skip(reason="This test is failing, needs more investigation")
def test_sidebias(
    env_name="NAltPerceptualDecisionMaking-v0",
    num_steps=100000,
    verbose=False,
    num_ch=3,
    margin=0.01,
    blk_dur=10,
    probs=None,
):
    """Test side_bias wrapper.

    The side-bias wrapper allows specifying the probabilities for each of the
    existing choices to be correct. These probabilities can varied in blocks of
    duration blk_dur.

    Parameters
    ----------
    env_name : ngym.env, optional
        enviroment to wrap. The default is 'NAltPerceptualDecisionMaking-v0'.
    num_steps : int, optional
        number of steps to run the environment (1000000)
    verbose : boolean, optional
        whether to print ground truth and block (False)
    num_ch : int, optional
        number of choices (3)
    blk_dur : int, optional
        duration (in trials) of the blocks (10)
    margin : float, optional
        margin allow when comparing provided and obtained ground truth
        probabilities (0.01)
    probs : list, optional
        ground truth probabilities for each block.
        For example: [(.005, .005, .99), (.005, .99, .005), (.99, .005, .005)],
        corresponds to 3 blocks with each of them giving 0.99 probabilitiy to
        ground truth 3, 2 and 1, respectively.

    Returns:
    -------
    None.

    """
    if probs is None:
        probs = [(0.005, 0.005, 0.99), (0.005, 0.99, 0.005), (0.99, 0.005, 0.005)]
    env_args["n_ch"] = num_ch
    env = gym.make(env_name, **env_args)
    env = SideBias(env, probs=probs, block_dur=blk_dur)
    env.reset()
    probs_mat = np.zeros((len(probs), num_ch))
    block = env.curr_block
    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)
        if info["new_trial"]:
            probs_mat[block, info["gt"] - 1] += 1
            block = env.curr_block
            if verbose:
                print("Ground truth", info["gt"])
                print("----------")
                print("Block", block)
        if terminated:
            env.reset()
    probs_mat = probs_mat / np.sum(probs_mat, axis=1)
    assert np.mean(np.abs(probs - probs_mat)) < margin, (
        "Probs provided " + str(probs) + " probs. obtained " + str(probs_mat)
    )
    print("-----")
    print("Side bias wrapper OK")


def test_passaction(
    env_name="PerceptualDecisionMaking-v0",
    num_steps=1000,
    verbose=True,
):
    """Test pass-action wrapper.

    TODO: explain wrapper
    Parameters
    ----------
    env_name : str, optional
        enviroment to wrap.. The default is 'PerceptualDecisionMaking-v0'.
    num_steps : int, optional
        number of steps to run the environment (1000)
    verbose : boolean, optional
        whether to print observation and action (False)

    Returns:
    -------
    None.

    """
    env = gym.make(env_name)
    env = PassAction(env)
    env.reset()
    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)
        assert obs[-1] == action, "Previous action is not part of observation"
        if verbose:
            print(obs)
            print(action)
            print("--------")

        if terminated:
            env.reset()


def test_passreward(
    env_name="PerceptualDecisionMaking-v0",
    num_steps=1000,
    verbose=False,
):
    """Test pass-reward wrapper.
    TODO: explain wrapper
    Parameters.
    ----------
    env_name : str, optional
        enviroment to wrap.. The default is 'PerceptualDecisionMaking-v0'.
    num_steps : int, optional
        number of steps to run the environment (1000)
    verbose : boolean, optional
        whether to print observation and reward (False)

    Returns:
    -------
    None.

    """
    env = gym.make(env_name)
    env = PassReward(env)
    obs, _ = env.reset()
    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)
        assert obs[-1] == rew, "Previous reward is not part of observation"
        if verbose:
            print(obs)
            print(rew)
            print("--------")
        if terminated:
            env.reset()


@pytest.mark.skip(reason="This test is failing, needs more investigation")
def test_reactiontime(
    env_name="PerceptualDecisionMaking-v0",
    num_steps=10000,
    urgency=-0.1,
    ths=None,
    verbose=True,
):
    """Test reaction-time wrapper.

    The reaction-time wrapper allows converting a fix duration task into a reaction
    time task. It also allows addding a fix (negative) quantity (urgency) to force
    the network to respond quickly.
    Parameters
    ----------
    env_name : str, optional
        enviroment to wrap.. The default is 'PerceptualDecisionMaking-v0'.
    num_steps : int, optional
        number of steps to run the environment (1000)
    urgency : float, optional
        float value added to the reward (-0.1)
    verbose : boolean, optional
        whether to print observation and reward (False)
    ths : list, optional
        list containing the threholds to make a decision ([-.5, .5])

    Returns:
    -------
    None.

    """
    if ths is None:
        ths = [-0.5, 0.5]
    env_args = {"timing": {"fixation": 100, "stimulus": 2000, "decision": 200}}
    env = gym.make(env_name, **env_args)
    env = ReactionTime(env, urgency=urgency)
    env.reset()
    if verbose:
        observations = []
        obs_cum_mat = []
        actions = []
        new_trials = []
        reward = []
    obs_cum = 0
    end_of_trial = False
    for _ in range(num_steps):
        if obs_cum > ths[1]:
            action = 1
        elif obs_cum < ths[0]:
            action = 2
        else:
            action = 0
        end_of_trial = action != 0
        obs, rew, terminated, truncated, info = env.step(action)
        if info["new_trial"]:
            step = 0
            obs_cum = 0
            end_of_trial = False
        else:
            step += 1
            assert not end_of_trial, "Trial still on after making a decision"
            obs_cum += obs[1] - obs[2]
        if verbose:
            observations.append(obs)
            actions.append(action)
            obs_cum_mat.append(obs_cum)
            new_trials.append(info["new_trial"])
            reward.append(rew)
    if verbose:
        observations = np.array(observations)
        _, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
        ax = ax.flatten()
        ax[0].imshow(observations.T, aspect="auto")
        ax[1].plot(actions, label="Actions")
        ax[1].plot(new_trials, "--", label="New trial")
        ax[1].set_xlim([-0.5, len(actions) - 0.5])
        ax[1].legend()
        ax[2].plot(obs_cum_mat, label="cum. observation")
        ax[2].plot([0, len(obs_cum_mat)], [ths[1], ths[1]], "--", label="upper th")
        ax[2].plot([0, len(obs_cum_mat)], [ths[0], ths[0]], "--", label="lower th")
        ax[2].set_xlim([-0.5, len(actions) - 0.5])
        ax[3].plot(reward, label="reward")
        ax[3].set_xlim([-0.5, len(actions) - 0.5])


@pytest.mark.skip(
    reason="VariableMapping is not implemented in the current version of neurogym",
)
def test_variablemapping(
    env="NAltConditionalVisuomotor-v0",
    verbose=True,
    mapp_ch_prob=0.05,
    min_mapp_dur=10,
    def_act=1,
    num_steps=2000,
    n_stims=4,
    n_ch=4,
    margin=2,
    sess_end_prob=0.01,
    min_sess_dur=20,
):
    """Test variable-mapping wrapper.
    TODO: explain wrapper
    Parameters.
    ----------
    env_name : str, optional
        enviroment to wrap.. The default is 'NAltConditionalVisuomotor-v0'.
    num_steps : int, optional
        number of steps to run the environment (1000)
    verbose : boolean, optional
        whether to print observation and reward (False)
    mapp_ch_prob : float, optional
        probability of mapping change (0.1)
    min_mapp_dur : int, optional
         minimum number of trials for a mapping block (3)
    sess_end_prob: float, optional,
        probability of session to finish (0.0025)
    min_sess_dur: int, optional
        minimum number of trials for session (5)
    def_act : int, optional
        default action for the agent, if None an action will be randomly chosen (1)
    n_stims : int, optional
        number of stims (10)
    n_ch : int, optional
        number of channels (4)
    margin : float, optional
        margin allowed when comparing actual and expected mean block durations (2)

    Returns:
    -------
    None.

    """
    env_args = {
        "n_stims": n_stims,
        "n_ch": n_ch,
        "timing": {"fixation": 100, "stimulus": 200, "delay": 200, "decision": 200},
    }

    env = gym.make(env, **env_args)
    env = VariableMapping(
        env,
        mapp_ch_prob=mapp_ch_prob,
        min_mapp_dur=min_mapp_dur,
        sess_end_prob=sess_end_prob,
        min_sess_dur=min_sess_dur,
    )
    env.reset()
    if verbose:
        observations = []
        reward = []
        actions = []
        gt = []
        new_trials = []
        mapping = []
        new_session = []
    prev_mapp = env.curr_mapping[env.trial["ground_truth"]] + 1
    stims = env.stims.flatten()
    for _ in range(num_steps):
        action = def_act or env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)
        if info["new_trial"]:
            mapping.append(info["mapping"])
            assert (action == prev_mapp and rew == 1.0) or action != prev_mapp
            prev_mapp = env.curr_mapping[env.trial["ground_truth"]] + 1
            if info["sess_end"]:
                new_session.append(1)
                assert (stims != env.stims.flatten()).any()
                stims = env.stims.flatten()
            else:
                new_session.append(0)
                assert (stims == env.stims.flatten()).all()
        if verbose:
            observations.append(obs)
            actions.append(action)
            reward.append(rew)
            new_trials.append(info["new_trial"])
            gt.append(info["gt"])
    mapping = [int(x.replace("-", "")) for x in mapping]
    mapp_ch = np.where(np.diff(mapping) != 0)[0]
    sess_ch = np.where(new_session)[0]
    if verbose:
        observations = np.array(observations)
        _, ax = plt.subplots(nrows=3, ncols=1, sharex=True)
        ax = ax.flatten()
        ax[0].imshow(observations.T, aspect="auto")
        ax[1].plot(actions, label="Actions")
        ax[1].plot(gt, "--", label="gt")
        ax[1].set_xlim([-0.5, len(actions) - 0.5])
        ax[1].legend()
        ax[2].plot(reward)
        end_of_trial = np.where(new_trials)[0]
        for ch in end_of_trial:
            ax[2].plot([ch, ch], [0, 1], "--c")
        for ch in mapp_ch:
            ax[2].plot([end_of_trial[ch], end_of_trial[ch]], [0, 1], "--k")
        for ch in sess_ch:
            ax[2].plot([end_of_trial[ch], end_of_trial[ch]], [0, 1], "--m")
        ax[2].set_xlim([-0.5, len(actions) - 0.5])
    sess_durs = np.diff(sess_ch)
    assert (sess_durs > min_sess_dur).all()
    mean_sess_dur = np.mean(sess_durs)
    exp_sess_durs = min_sess_dur + 1 / sess_end_prob
    assert np.abs(mean_sess_dur - exp_sess_durs) < margin, (
        "Mean sess. dur.: " + str(mean_sess_dur) + ", expected: " + str(1 / sess_end_prob)
    )
    mapp_blck_durs = np.diff(mapp_ch)
    assert (mapp_blck_durs > min_mapp_dur).all()
    mean_durs = np.mean(mapp_blck_durs)
    exp_durs = min_mapp_dur + 1 / mapp_ch_prob
    assert np.abs(mean_durs - exp_durs) < margin, (
        "Mean mapp. block durations: " + str(mean_durs) + ", expected: " + str(exp_durs)
    )
    sys.exit()


@pytest.mark.skip(reason="This test is failing, needs more investigation")
def test_noise(
    env="PerceptualDecisionMaking-v0",
    margin=0.01,
    perf_th=0.7,
    num_steps=100000,
    verbose=True,
):
    """Test noise wrapper.

    The noise wrapper allows adding noise to the full observation received by the
    network. It also offers the option of fixxing a specific target performance
    that the wrapper will assure by modulating the magnitude of the noise added.
    Parameters
    ----------
    env_name : str, optional
        enviroment to wrap.. The default is 'PerceptualDecisionMaking-v0'.
    num_steps : int, optional
        number of steps to run the environment (1000)
    verbose : boolean, optional
        whether to print observation and reward (False)
    margin : float, optional
        margin allowed when comparing actual and expected performances (0.01)
    perf_th : float, optional
        target performance for the noise wrapper (0.7)

    Returns:
    -------
    None.

    """
    env_args = {"timing": {"fixation": 100, "stimulus": 200, "decision": 200}}
    env = gym.make(env, **env_args)
    env = Noise(env, perf_th=perf_th)
    env.reset()
    perf = []
    std_mat = []
    std_noise = 0
    for _ in range(num_steps):
        action = env.action_space.sample() if np.random.rand() < std_noise else env.gt_now
        obs, rew, terminated, truncated, info = env.step(action)
        if "std_noise" in info:
            std_noise = info["std_noise"]
        if verbose and info["new_trial"]:
            perf.append(info["performance"])
            std_mat.append(std_noise)
        if terminated:
            env.reset()
    actual_perf = np.mean(perf[-5000:])
    if verbose:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot([0, len(perf)], [perf_th, perf_th], "--")
        plt.plot(np.convolve(perf, np.ones((100,)) / 100, mode="valid"))
        plt.subplot(2, 1, 2)
        plt.plot(std_mat)
    assert np.abs(actual_perf - perf_th) < margin, (
        "Actual performance: " + str(actual_perf) + ", expected: " + str(perf_th)
    )


@pytest.mark.skip(
    reason="TimeOut is not implemented in the current version of neurogym",
)
def test_timeout(
    env="NAltPerceptualDecisionMaking-v0",
    time_out=500,
    num_steps=100,
    verbose=True,
):
    env_args = {
        "n_ch": 2,
        "timing": {"fixation": 100, "stimulus": 200, "decision": 200},
    }
    env = gym.make(env, **env_args)
    env = TimeOut(env, time_out=time_out)
    env.reset()
    reward = []
    observations = []
    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)
        if verbose:
            reward.append(rew)
            observations.append(obs)
        if terminated:
            env.reset()
    if verbose:
        observations = np.array(observations)
        _, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax = ax.flatten()
        ax[0].imshow(observations.T, aspect="auto")
        ax[1].plot(reward, "--", label="reward")
        ax[1].set_xlim([-0.5, len(reward) - 0.5])
        ax[1].legend()


@pytest.mark.skip(
    reason="CatchTrials is not implemented in the current version of neurogym",
)
def test_catchtrials(
    env_name,
    num_steps=10000,
    verbose=False,
    catch_prob=0.1,
    alt_rew=0,
):
    env = gym.make(env_name)
    env = CatchTrials(env, catch_prob=catch_prob, alt_rew=alt_rew)
    env.reset()
    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)
        if info["new_trial"] and verbose:
            print("Perfomance", (info["gt"] - action) < 0.00001)
            print("catch-trial", info["catch_trial"])
            print("Reward", rew)
            print("-------------")
        if terminated:
            env.reset()


@pytest.mark.skip(
    reason="TrialHistory and Variable_nch are not implemented in the current version of neurogym",
)
def test_trialhist_and_variable_nch(
    env_name,
    num_steps=100000,
    probs=0.8,
    num_blocks=2,
    verbose=False,
    num_ch=4,
    variable_nch=False,
):
    env = gym.make(env_name, n_ch=num_ch)
    env = TrialHistory(env, probs=probs, block_dur=200, num_blocks=num_blocks)
    if variable_nch:
        env = Variable_nch(env, block_nch=1000, blocks_probs=[0.1, 0.45, 0.45])
        transitions = np.zeros((num_ch - 1, num_blocks, num_ch, num_ch))
    else:
        transitions = np.zeros((1, num_blocks, num_ch, num_ch))
    env.reset()
    blk = []
    gt = []
    nch = []
    prev_gt = 1
    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)
        if terminated:
            env.reset()
        if info["new_trial"] and verbose:
            blk.append(info["curr_block"])
            gt.append(info["gt"])
            if variable_nch:
                nch.append(info["nch"])
                if len(nch) > 1 and nch[-1] == nch[-2] and blk[-1] == blk[-2]:
                    transitions[
                        info["nch"] - 2,
                        info["curr_block"],
                        prev_gt,
                        info["gt"] - 1,
                    ] += 1
            else:
                nch.append(num_ch)
                transitions[0, info["curr_block"], prev_gt, info["gt"] - 1] += 1
            prev_gt = info["gt"] - 1
    if verbose:
        _, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax[0].plot(blk[:20000], "-+")
        ax[0].plot(nch[:20000], "-+")
        ax[1].plot(gt[:20000], "-+")
        ch_mat = np.unique(nch)
        _, ax = plt.subplots(nrows=num_blocks, ncols=len(ch_mat))
        for ind_ch, _ in enumerate(ch_mat):
            for ind_blk in range(num_blocks):
                norm_counts = transitions[ind_ch, ind_blk, :, :]
                nxt_tr_counts = np.sum(norm_counts, axis=1).reshape((-1, 1))
                norm_counts = norm_counts / nxt_tr_counts
                ax[ind_blk][ind_ch].imshow(norm_counts)


@pytest.mark.skip(
    reason="TTLPulse is not implemented in the current version of neurogym",
)
def test_ttlpulse(env_name, num_steps=10000, verbose=False, **envArgs):
    env = gym.make(env_name, **envArgs)
    env = TTLPulse(env, periods=[["stimulus"], ["decision"]])
    env.reset()
    obs_mat = []
    signals = []
    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)
        if verbose:
            obs_mat.append(obs)
            signals.append([info["signal_0"], info["signal_1"]])
            print("--------")

        if terminated:
            env.reset()
    if verbose:
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.title("Observations")
        plt.imshow(np.array(obs_mat).T, aspect="auto")
        plt.subplot(2, 1, 2)
        plt.title("Pulses")
        plt.plot(signals)
        plt.xlim([-0.5, num_steps - 0.5])


@pytest.mark.skip(
    reason="TransferLearning is not implemented in the current version of neurogym",
)
def test_transferLearning(num_steps=10000, verbose=False, **envArgs):
    task = "GoNogo-v0"
    kwargs = {
        "dt": 100,
        "timing": {"fixation": 0, "stimulus": 100, "resp_delay": 100, "decision": 100},
    }
    env1 = gym.make(task, **kwargs)
    task = "PerceptualDecisionMaking-v0"
    kwargs = {"dt": 100, "timing": {"fixation": 100, "stimulus": 100, "decision": 100}}
    env2 = gym.make(task, **kwargs)
    env = TransferLearning([env1, env2], num_tr_per_task=[30], task_cue=True)

    env.reset()
    obs_mat = []
    signals = []
    rew_mat = []
    action_mat = []
    for _ in range(num_steps):
        action = 1
        obs, rew, terminated, truncated, info = env.step(action)
        if verbose:
            action_mat.append(action)
            rew_mat.append(rew)
            obs_mat.append(obs)
            signals.append([info["task"]])

        if terminated:
            env.reset()
    if verbose:
        plt.figure()
        plt.subplot(4, 1, 1)
        plt.title("Observations")
        plt.imshow(np.array(obs_mat).T, aspect="auto")
        plt.subplot(4, 1, 4)
        plt.title("Pulses")
        plt.plot(signals)
        plt.xlim([-0.5, num_steps - 0.5])
        plt.subplot(4, 1, 3)
        plt.title("Actions")
        plt.plot(action_mat)
        plt.xlim([-0.5, num_steps - 0.5])
        plt.subplot(4, 1, 2)
        plt.title("Reward")
        plt.plot(rew_mat)
        plt.xlim([-0.5, num_steps - 0.5])


@pytest.mark.skip(
    reason="Combine is not implemented in the current version of neurogym",
)
def test_combine(num_steps=10000, verbose=False, **envArgs):
    task = "GoNogo-v0"
    kwargs = {
        "dt": 100,
        "timing": {"fixation": 0, "stimulus": 100, "resp_delay": 100, "decision": 100},
    }
    env1 = gym.make(task, **kwargs)
    task = "DelayPairedAssociation-v0"
    kwargs = {
        "dt": 100,
        "timing": {
            "fixation": 0,
            "stim1": 100,
            "delay_btw_stim": 500,
            "stim2": 100,
            "delay_aft_stim": 100,
            "decision": 200,
        },
    }
    env2 = gym.make(task, **kwargs)
    env = Combine(
        env=env1,
        distractor=env2,
        delay=100,
        dt=100,
        mix=(0.3, 0.3, 0.4),
        share_action_space=True,
        defaults=[0, 0],
        trial_cue=True,
    )

    env.reset()
    obs_mat = []
    config_mat = []
    rew_mat = []
    action_mat = []
    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)
        if verbose:
            action_mat.append(action)
            rew_mat.append(rew)
            obs_mat.append(obs)
            config_mat.append(info["task_type"])

        if terminated:
            env.reset()
    if verbose:
        plt.figure()
        plt.subplot(4, 1, 1)
        plt.title("Observations")
        plt.imshow(np.array(obs_mat).T, aspect="auto")
        plt.subplot(4, 1, 4)
        plt.title("Trial configuration")
        plt.plot(config_mat)
        plt.xlim([-0.5, num_steps - 0.5])
        plt.subplot(4, 1, 3)
        plt.title("Actions")
        plt.plot(action_mat)
        plt.xlim([-0.5, num_steps - 0.5])
        plt.subplot(4, 1, 2)
        plt.title("Reward")
        plt.plot(rew_mat)
        plt.xlim([-0.5, num_steps - 0.5])


@pytest.mark.skip(
    reason="Identity is not implemented in the current version of neurogym",
)
def test_identity(env_name, num_steps=10000, **envArgs):
    env = gym.make(env_name, **envArgs)
    env = Identity(env)
    env = Identity(env, id_="1")
    env.reset()
    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)
        if terminated:
            env.reset()


@pytest.mark.skip(reason="This test is failing, needs more investigation")
def test_all(test_fn):
    """Test speed of all experiments."""
    success_count = 0
    total_count = 0
    for env_name in sorted(ngym.all_envs()):
        total_count += 1
        print(f"Running env: {env_name:s} Wrapped with SideBias")
        try:
            test_fn(env_name)
            print("Success")
            success_count += 1
        except BaseException as e:  # noqa: BLE001 # FIXME: unclear which error is expected here.
            print(f"Failure at running env: {env_name:s}")
            print(e)
        print()

    print(f"Success {success_count:d}/{total_count:d} envs")


@pytest.mark.skip(
    reason="TrialHistoryEvolution and Variable_nch are not implemented in the current version of neurogym",
)
def test_concat_wrpprs_th_vch_pssr_pssa(
    env_name,
    num_steps=100000,
    probs=0.8,
    num_blocks=16,
    verbose=False,
    num_ch=8,
    env_args=None,
):
    if env_args is None:
        env_args = {}
    env_args["n_ch"] = num_ch
    env_args["zero_irrelevant_stim"] = True
    env_args["ob_histblock"] = True
    env = gym.make(env_name, **env_args)
    env = TrialHistoryEvolution(
        env,
        probs=probs,
        ctx_ch_prob=0.005,
        predef_tr_mats=True,
        balanced_probs=True,
        num_contexts=1,
    )
    env = Variable_nch(env, block_nch=5000000000, prob_12=0.05, sorted_ch=True)
    transitions = np.zeros((num_blocks, num_ch, num_ch))
    env = PassReward(env)
    env = PassAction(env)
    env.reset()
    num_tr_blks = np.zeros((num_blocks,))
    blk_id = []
    s_chs = []
    blk = []
    blk_stp = []
    gt = []
    nch = []
    obs_mat = []
    prev_gt = 1
    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)
        obs_mat.append(obs)
        blk_stp.append(info["curr_block"])
        if terminated:
            env.reset()
        if info["new_trial"] and verbose:
            blk.append(info["curr_block"])
            gt.append(info["gt"])
            sel_chs = list(info["sel_chs"].replace("-", ""))
            sel_chs = [int(x) - 1 for x in sel_chs]
            blk_id, indx = check_blk_id(blk_id, info["curr_block"], num_blocks)
            s_chs.append(info["sel_chs"])
            nch.append(info["nch"])
            if len(nch) > 2 and 2 * [nch[-1]] == nch[-3:-1] and 2 * [blk[-1]] == blk[-3:-1] and indx != -1:
                num_tr_blks[indx] += 1
                transitions[indx, prev_gt, info["gt"] - 1] += 1
                if prev_gt > info["nch"] or info["gt"] - 1 > info["nch"]:
                    pass
            prev_gt = info["gt"] - 1
    if verbose:
        print(blk_id)
        sel_choices, counts = np.unique(s_chs, return_counts=1)
        print("\nSelected choices and frequencies:")
        print(sel_choices)
        print(counts / np.sum(counts))
        tr_blks, counts = np.unique(
            np.array(blk)[np.array(s_chs) == "1-2"],
            return_counts=1,
        )
        print("\n2AFC task transition matrices and frequencies:")
        print(tr_blks)
        print(counts / np.sum(counts))
        _, ax = plt.subplots(nrows=1, ncols=1)
        obs_mat = np.array(obs_mat)
        ax.imshow(obs_mat[10000:20000, :].T, aspect="auto")
        _, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
        blk_int = [int(x.replace("-", "")) for x in blk]
        ax[0].plot(
            np.array(blk_int[:20000]) / (10 ** (num_ch - 1)),
            "-+",
            label="tr-blck",
        )
        ax[0].plot(nch[:20000], "-+", label="num choices")
        ax[1].plot(gt[:20000], "-+", label="correct side")
        ax[1].set_xlabel("Trials")
        ax[0].legend()
        ax[1].legend()
        num_cols_rows = int(np.sqrt(num_blocks))
        _, ax1 = plt.subplots(ncols=num_cols_rows, nrows=num_cols_rows)
        ax1 = ax1.flatten()
        _, ax2 = plt.subplots(ncols=num_cols_rows, nrows=num_cols_rows)
        ax2 = ax2.flatten()
        for ind_blk in range(len(blk_id)):
            norm_counts = transitions[ind_blk, :, :]
            ax1[ind_blk].imshow(norm_counts)
            ax1[ind_blk].set_title(
                str(blk_id[ind_blk]) + " (N=" + str(num_tr_blks[ind_blk]) + ")",
                fontsize=6,
            )
            nxt_tr_counts = np.sum(norm_counts, axis=1).reshape((-1, 1))
            norm_counts = norm_counts / nxt_tr_counts
            ax2[ind_blk].imshow(norm_counts)
            ax2[ind_blk].set_title(
                str(blk_id[ind_blk]) + " (N=" + str(num_tr_blks[ind_blk]) + ")",
                fontsize=6,
            )
    return {
        "transitions": transitions,
        "blk": blk,
        "blk_id": blk_id,
        "gt": gt,
        "nch": nch,
        "s_ch": s_chs,
        "obs_mat": obs_mat,
        "blk_stp": blk_stp,
    }


def check_blk_id(blk_id_mat, curr_blk, num_blk):
    # translate transitions t.i.a. selected choices
    if curr_blk in blk_id_mat:
        return blk_id_mat, np.argwhere(np.array(blk_id_mat) == curr_blk)
    if len(blk_id_mat) < num_blk:
        blk_id_mat.append(curr_blk)
        return blk_id_mat, len(blk_id_mat) - 1
    return blk_id_mat, -1


@pytest.mark.skip(
    reason="TrialHistoryEvolution is not implemented in the current version of neurogym",
)
def test_trialhistEv(
    env_name,
    num_steps=10000,
    probs=0.8,
    num_blocks=2,
    verbose=True,
    num_ch=4,
):
    env = gym.make(env_name, n_ch=num_ch)
    env = TrialHistoryEvolution(
        env,
        probs=probs,
        ctx_dur=200,
        death_prob=0.001,
        num_contexts=num_blocks,
        fix_2AFC=True,
        balanced_probs=True,
    )
    transitions = []
    env.reset()
    num_tr = 0
    for _ in range(num_steps):
        action = env.action_space.sample()
        obs, rew, terminated, truncated, info = env.step(action)
        if terminated:
            env.reset()
        if info["new_trial"] and verbose:
            num_tr += 1
            transitions.append(
                np.array([np.where(x == 0.8)[0][0] for x in env.curr_tr_mat[0, :, :]]),
            )
        if info["new_generation"] and verbose:
            print("New generation")
            print(num_tr)
    plt.figure()
    plt.imshow(np.array(transitions), aspect="auto")


if __name__ == "__main__":
    plt.close("all")
    env_args = {
        "stim_scale": 10,
        "timing": {"fixation": 100, "stimulus": 200, "decision": 200},
    }
    # test_identity('Null-v0', num_steps=5) # noqa: ERA001
    test_reactiontime()
    sys.exit()
    test_timeout()
    test_noise()
    sys.exit()
    test_variablemapping()
    sys.exit()
    test_sidebias()
    test_passreward()
    test_passaction()

    test_ttlpulse("PerceptualDecisionMaking-v0", num_steps=20, verbose=True, **env_args)
    test_catchtrials(
        "PerceptualDecisionMaking-v0",
        num_steps=10000,
        verbose=True,
        catch_prob=0.5,
        alt_rew=0,
    )
    test_transferLearning(num_steps=200, verbose=True)
    test_combine(num_steps=200, verbose=True)
    test_trialhist_and_variable_nch(
        "NAltPerceptualDecisionMaking-v0",
        num_steps=1000000,
        verbose=True,
        probs=0.99,
        num_blocks=3,
    )

    # test_trialhistEv('NAltPerceptualDecisionMaking-v0', num_steps=100000,
    #                  probs=0.8, num_blocks=3, verbose=True, num_ch=8)
    # data = test_concat_wrpprs_th_vch_pssr_pssa('NAltPerceptualDecisionMaking-v0',
    #                                            num_steps=20000, verbose=True,
    #                                            probs=0.99, num_blocks=16,
    #                                            env_args=env_args)
