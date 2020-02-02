"""Utilities for data."""

import numpy as np
import gym


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

    def __init__(self, env_name, env_kwargs=None,
                 batch_size=1, seq_len=None, cache_len=None):
        if env_kwargs is None:
            env_kwargs = {}
        self.envs = [gym.make(env_name, **env_kwargs) for _ in range(batch_size)]
        for env in self.envs:
            env.reset()
        env = self.envs[0]
        self.env = env
        self.batch_size = batch_size

        if seq_len is None:
            # TODO: infer sequence length from task
            seq_len = 1000

        observation_shape = env.observation_space.shape
        action_shape = env.action_space.shape
        if len(action_shape) == 0:
            self._expand_action = True
        else:
            self._expand_action = False

        if cache_len is None:
            # Infer cache len
            cache_len = 1e7
            cache_len /= (np.prod(observation_shape) + np.prod(action_shape))
            cache_len /= batch_size
        cache_len = int((cache_len // seq_len) * seq_len)

        self.seq_len = seq_len
        self.inputs_shape = [batch_size, seq_len] + list(observation_shape)
        self.target_shape = [batch_size, seq_len] + list(action_shape)

        self._cache_len = cache_len
        self._cache_inputs_shape = [batch_size, cache_len] + list(observation_shape)
        self._cache_target_shape = [batch_size, cache_len] + list(action_shape)

        self._cache()

    def _cache(self):
        self._inputs = np.zeros(self._cache_inputs_shape)
        self._target = np.zeros(self._cache_target_shape)

        for i in range(self.batch_size):
            env = self.envs[i]
            seq_start = 0
            seq_end = 0
            while seq_end < self._cache_len:
                env.new_trial()
                obs, gt = env.obs, env.gt
                seq_len = obs.shape[0]
                seq_end = seq_start + seq_len
                if seq_end > self._cache_len:
                    seq_end = self._cache_len
                    seq_len = seq_end - seq_start
                self._inputs[i, seq_start:seq_end, ...] = obs[:seq_len]
                self._target[i, seq_start:seq_end, ...] = gt[:seq_len]
                seq_start = seq_end

        self._seq_start = 0

        if self._expand_action:
            self._target = self._target[..., np.newaxis]

    def __iter__(self):
        return self

    def __call__(self, *args, **kwargs):
        return self.__next__()

    def __next__(self):
        self._seq_end = self._seq_start + self.seq_len
        if self._seq_end > self._cache_len:
            self._cache()

        return (self._inputs[:, self._seq_start:self._seq_end, ...],
                self._target[:, self._seq_start:self._seq_end, ...])


if __name__ == '__main__':
    import neurogym as ngym
    dataset = ngym.Dataset(
        'PerceptualDecisionMaking-v0', env_kwargs={'dt': 20}, batch_size=32)
    for i in range(100):
        inputs, target = dataset()
    print(inputs.shape)
    print(target.shape)


