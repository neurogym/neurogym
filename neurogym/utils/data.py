"""Utilities for data."""

import copy

import gymnasium as gym
import numpy as np


class Dataset:
    """Make an environment into an iterable dataset for supervised learning.

    Create an iterator that at each call returns
        inputs: numpy array (sequence_length, batch_size, input_units)
        target: numpy array (sequence_length, batch_size, output_units)

    Args:
        env: str for env id or gym.Env objects
        env_kwargs: dict, additional kwargs for environment, if env is str
        batch_size: int, batch size
        seq_len: int, sequence length
        max_batch: int, maximum number of batch for iterator, default infinite
        batch_first: bool, if True, return (batch, seq_len, n_units), default False
        cache_len: int, default length of caching
    """

    def __init__(
        self,
        env,
        env_kwargs=None,
        batch_size=1,
        seq_len=None,
        max_batch=np.inf,
        batch_first=False,
        cache_len=None,
    ) -> None:
        if not isinstance(env, str | gym.Env):
            msg = f"{type(env)=} must be `gym.Env` or `str`."
            raise TypeError(msg)
        if isinstance(env, gym.Env):
            self.envs = [copy.deepcopy(env) for _ in range(batch_size)]
        else:
            if env_kwargs is None:
                env_kwargs = {}
            self.envs = [gym.make(env, **env_kwargs) for _ in range(batch_size)]
        for env_ in self.envs:
            env_.reset()
        self.seed()

        env = self.envs[0]
        self.env = env
        self.batch_size = batch_size
        self.batch_first = batch_first

        if seq_len is None:
            # TODO: infer sequence length from task
            seq_len = 1000

        obs_shape = env.observation_space.shape
        action_shape = env.action_space.shape
        if obs_shape is None or action_shape is None:
            msg = "The observation and action spaces must have a shape."
            raise ValueError(msg)

        self._expand_action = len(action_shape) == 0

        if cache_len is None:
            # Infer cache len
            cache_len = 1e5  # Probably too low
            cache_len /= np.prod(obs_shape) + np.prod(action_shape)
            cache_len /= batch_size
        cache_len = int((1 + (cache_len // seq_len)) * seq_len)

        self.seq_len = seq_len
        self._cache_len = cache_len

        if batch_first:
            shape1, shape2 = [batch_size, seq_len], [batch_size, cache_len]
        else:
            shape1, shape2 = [seq_len, batch_size], [cache_len, batch_size]

        self.inputs_shape = shape1 + list(obs_shape)
        self.target_shape = shape1 + list(action_shape)
        self._cache_inputs_shape = shape2 + list(obs_shape)
        self._cache_target_shape = shape2 + list(action_shape)

        self._inputs = np.zeros(
            self._cache_inputs_shape,
            dtype=env.observation_space.dtype,
        )
        self._target = np.zeros(self._cache_target_shape, dtype=env.action_space.dtype)

        self._cache()

        self._i_batch = 0
        self.max_batch = max_batch

    def _cache(self, **kwargs) -> None:
        for i in range(self.batch_size):
            env = self.envs[i]
            seq_start = 0
            seq_end = 0
            while seq_end < self._cache_len:
                # TODO: Right now this only works for env with new_trial
                env.new_trial(**kwargs)  # type: ignore[attr-defined]
                ob, gt = env.ob, env.gt  # type: ignore[attr-defined]
                seq_len = ob.shape[0]
                seq_end = seq_start + seq_len
                if seq_end > self._cache_len:
                    seq_end = self._cache_len
                    seq_len = seq_end - seq_start
                if self.batch_first:
                    self._inputs[i, seq_start:seq_end, ...] = ob[:seq_len]
                    self._target[i, seq_start:seq_end, ...] = gt[:seq_len]
                else:
                    self._inputs[seq_start:seq_end, i, ...] = ob[:seq_len]
                    self._target[seq_start:seq_end, i, ...] = gt[:seq_len]
                seq_start = seq_end

        self._seq_start = 0
        self._seq_end = self._seq_start + self.seq_len

    def __iter__(self):
        return self

    def __call__(self, *args, **kwargs):
        return self.__next__(**kwargs)

    def __next__(self, **kwargs):
        self._i_batch += 1
        if self._i_batch > self.max_batch:
            self._i_batch = 0
            raise StopIteration

        self._seq_end = self._seq_start + self.seq_len

        if self._seq_end >= self._cache_len:
            self._cache(**kwargs)

        if self.batch_first:
            inputs = self._inputs[:, self._seq_start : self._seq_end, ...]
            target = self._target[:, self._seq_start : self._seq_end, ...]
        else:
            inputs = self._inputs[self._seq_start : self._seq_end]
            target = self._target[self._seq_start : self._seq_end]

        self._seq_start = self._seq_end
        return inputs, target

    def seed(self, seed=None) -> None:
        for i, env in enumerate(self.envs):
            if seed is None:
                env.seed(seed)  # type: ignore[attr-defined]
            else:
                env.seed(seed + i)  # type: ignore[attr-defined]


if __name__ == "__main__":
    import neurogym as ngym

    dataset = ngym.Dataset(
        "PerceptualDecisionMaking-v0",
        env_kwargs={"dt": 100},
        batch_size=32,
        seq_len=40,
    )
    inputs_list = []
    for _ in range(2):
        inputs, target = dataset()
        inputs_list.append(inputs)
