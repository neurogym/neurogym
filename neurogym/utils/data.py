"""Utilities for data."""

import numpy as np


class Dataset(object):
    """Make an environment into an iterable dataset.

    Create an iterator that generates
        inputs: numpy array (batch_size, sequence_length, input_units)
        target: numpy array (batch_size, sequence_length, output_units)

    There are two modes:
        When single_trial=True, each batch contains only a single trial,
        seq_len is either provided by user or is the maximum across all trials
        When single_trial=False, seq_len must be provided, and env will be run
        until the total length is seq_len
    """

    def __init__(self, env, batch_size=1, single_trial=True, seq_len=None):
        self.env = env
        self.batch_size = batch_size
        self.single_trial = single_trial

        if not single_trial and seq_len is None:
            raise ValueError('seq_len can not be None when single_trial is False')

        if seq_len is None:
            # TODO: infer sequence length from task
            seq_len = 100

        self.seq_len = seq_len
        self.inputs_shape = [batch_size, seq_len] + list(self.env.observation_space.shape)
        self.target_shape = [batch_size, seq_len] + list(self.env.action_space.shape)

        if batch_size > 1:
            raise ValueError('Larger batch size not implemented yet')

    def __iter__(self):
        return self

    def __call__(self, *args, **kwargs):
        return self.__next__()

    def __next__(self):
        self.env.new_trial()
        obs, gt = self.env.obs, self.env.gt
        min_seq_len = np.min((obs.shape[0], self.seq_len))
        inputs = np.zeros(self.inputs_shape)
        target = np.zeros(self.target_shape)
        inputs[0, :min_seq_len, ...] = obs[:min_seq_len]
        target[0, :min_seq_len, ...] = gt[:min_seq_len]

        return inputs, target


def get_dataset_for_SL(env, rollout, n_tr, n_steps, obs_size,
                       act_size, seed=None):
    env.seed(seed)
    env.reset()
    # TODO: this assumes 1-D observations
    samples = np.empty((n_steps, obs_size))
    target = np.empty((n_steps, act_size))

    count_stps = 0
    for tr in range(n_tr):
        obs = env.obs
        gt = env.gt
        samples[count_stps:count_stps+obs.shape[0], :] = obs
        target[count_stps:count_stps+gt.shape[0], :] = np.eye(act_size)[gt]
        count_stps += obs.shape[0]
        assert obs.shape[0] == gt.shape[0]
        env.new_trial()

    samples = samples[:count_stps, :]
    target = target[:count_stps, :]
    samples = samples.reshape((-1, rollout, obs_size))
    target = target.reshape((-1, rollout, act_size))
    return samples, target, env


if __name__ == '__main__':
    from neurogym.envs.perceptualdecisionmaking import PerceptualDecisionMaking
    env = PerceptualDecisionMaking()
    dataset = Dataset(env)
    for i in range(10):
        inputs, target = dataset()
        if i > 10:
            break


