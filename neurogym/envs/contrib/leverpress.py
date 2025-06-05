from typing import NoReturn

import gymnasium as gym
import numpy as np
from gymnasium import logger, spaces
from gymnasium.utils import seeding

from neurogym.core import InvalidOperationError


class LeverPress(gym.Env):
    """Lever pressing environment where a cue signals the sequence start."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}  # noqa: RUF012

    def __init__(self) -> None:
        """Lever pressing environment where a cue signals the sequence start."""
        high = np.array([1])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.signal_prob = 0.5  # probability of signaling a sequence
        self.n_press = 4  # number of consecutive pressed needed
        self.reward_seq_complete = 10.0  # reward when sequence completed

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        if not self.action_space.contains(action):
            msg = f"{action=!r} ({type(action)}) not found in action_space."
            raise InvalidOperationError(msg)
        state = self.state

        reward = 0.0
        if state == 0:
            if action == 0:
                if self.np_random.rand() < self.signal_prob:
                    state = self.n_press
            else:
                reward = -1.0
        elif action == 1:
            # reducing state to 0
            state -= 1
            if state == 0:
                # this transition is rewarded
                reward = self.reward_seq_complete
        else:
            # if not pressing, then move directly to 0
            state = 0
        signal = [float(state == self.n_press)]

        self.state = state
        terminated = False
        truncated = False

        if not terminated:
            pass
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned terminated = True.\n",
                    "You should always call 'reset()' once you receive 'terminated = True'.",
                    "-- any further steps are undefined behavior.",
                )
            self.steps_beyond_done += 1

        return np.array(signal), reward, terminated, truncated, {}

    @property
    def optimal_reward(self):
        """Optimal reward possible for each step on average."""
        p = self.signal_prob
        ns = np.arange(100)
        r = 1.0 / np.sum((ns + self.n_press + 1) * p * (1 - p) ** ns)
        r *= self.reward_seq_complete
        return r

    def reset(self):
        self.state = 0
        signal = [0]
        self.steps_beyond_done = None
        return np.array(signal), {}

    def render(self, mode="human") -> NoReturn:
        raise NotImplementedError

    def close(self) -> None:
        if self.viewer:
            self.viewer.close()


class LeverPressWithPoke(gym.Env):
    """Lever press but obtain reward through poking.

    Observation:
        0: constant 1

    Actions:
        0: poking reward port
        1: pressing
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}  # noqa: RUF012

    def __init__(self) -> None:
        high = np.array([1])
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.n_press = 4  # number of consecutive pressed needed
        self.reward_seq_complete = 1.0  # reward when sequence completed

        self.seed()
        self.viewer = None
        self.state = self.n_press  # state is the number of presses remaining
        self.observe_state = False  # TEMPORARYLY TRUE

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        if not self.action_space.contains(action):
            msg = f"{action=!r} ({type(action)}) not found in action_space."
            raise InvalidOperationError(msg)
        state = self.state

        reward = 0.0
        if action == 0:
            # poke
            if state <= 0:
                # reward and reset state
                reward = self.reward_seq_complete
                state = self.n_press
        elif action == 1:
            # press
            state -= 1
            state = max(0, state)
        else:
            msg = f"{action=} must be 0 or 1."
            raise ValueError(msg)

        self.state = state
        terminated = False
        truncated = False

        if not terminated:
            pass
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned terminated = True.\n",
                    "You should always call 'reset()' once you receive 'terminated = True'.",
                    "-- any further steps are undefined behavior.",
                )
            self.steps_beyond_done += 1

        obs = np.array([self.state]) if self.observe_state else np.array([1.0])

        return obs, reward, terminated, truncated, {}

    @property
    def optimal_reward(self):
        """Optimal reward possible for each step on average."""
        return self.reward_seq_complete / (1.0 + self.n_press)

    @property
    def optimal_chance_reward(self):
        """Optimal reward if agent chooses press and poking randomly."""
        # optimal random p when state 0 --> N is the only rewarding transition
        p_opt = np.sqrt(self.n_press) / (1 + np.sqrt(self.n_press))
        p = p_opt
        r = p * (1 - p) / (self.n_press * (1 - p) + p)
        r *= self.reward_seq_complete
        return r

    def reset(self):
        self.state = self.n_press
        self.steps_beyond_done = None
        if self.observe_state:
            return np.array([self.state]), {}
        return np.array([1.0]), {}

    def render(self, mode="human") -> NoReturn:
        raise NotImplementedError

    def close(self) -> None:
        if self.viewer:
            self.viewer.close()


class LeverPressWithPokeRest(gym.Env):
    """Lever press but obtain reward through poking.

    Observation:
        0: thirsty level, ranging from 0 to 1, will scale reward obtained

    Actions:
        0: not do anything
        1: pressing
        2: poking reward port
    """

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}  # noqa: RUF012

    def __init__(self) -> None:
        high = np.array([1])
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.n_press = 4  # number of consecutive pressed needed
        self.reward_seq_complete = 2.0  # reward when sequence completed

        self.seed()
        self.viewer = None
        self.state = self.n_press  # state is the number of presses remaining
        self.observe_state = False  # TEMPORARYLY TRUE

        self.thirst_state = 1
        self.effort = -0.1

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        if not self.action_space.contains(action):
            msg = f"{action=!r} ({type(action)}) not found in action_space."
            raise InvalidOperationError(msg)
        state = self.state

        if action == 0:
            reward = self.effort
            # poke
            if state <= 0:
                # reward and reset state
                reward += self.reward_seq_complete * self.thirst
                state = self.n_press
                self.thirst_state = -8
        elif action == 1:
            # press
            reward = self.effort
            state -= 1
            state = max(0, state)
        elif action == 2:
            # rest
            reward = 0.0
        else:
            msg = f"{action=} must be 0 or 1 or 2."
            raise ValueError

        self.thirst_state += self.np_random.rand() * 0.4 + 0.8
        self.thirst = np.clip(self.thirst_state, 0.0, 1.0)
        self.state = state
        terminated = False
        truncated = False

        if not terminated:
            pass
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned terminated = True.\n",
                    "You should always call 'reset()' once you receive 'terminated = True'.",
                    "-- any further steps are undefined behavior.",
                )
            self.steps_beyond_done += 1

        obs = np.array([self.state]) if self.observe_state else np.array([self.thirst])

        return obs, reward, terminated, truncated, {}

    @property
    def optimal_reward(self) -> int:
        # TODO: Remain to be updated
        """Optimal reward possible for each step on average."""
        return 0

    @property
    def optimal_chance_reward(self) -> int:
        """Optimal reward if agent chooses press and poking randomly."""
        # TODO: Remain to be updated
        return 0

    def reset(self):
        self.state = self.n_press
        self.steps_beyond_done = None
        self.thirst_state = 1
        self.thirst = np.clip(self.thirst_state, 0.0, 1.0)
        if self.observe_state:
            return np.array([self.state]), {}
        return np.array([self.thirst]), {}

    def render(self, mode="human") -> NoReturn:
        raise NotImplementedError

    def close(self) -> None:
        if self.viewer:
            self.viewer.close()


class ContextSwitch(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}  # noqa: RUF012

    def __init__(self) -> None:
        high = np.array([1, 1])
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.context = 0  # state is the number of presses remaining
        self.p_switch = 0.1

        self.ob2action_context1 = {0: 0, 1: 1}
        self.ob2action_context2 = {0: 1, 1: 0}
        self.ob = 0

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, action):
        if not self.action_space.contains(action):
            msg = f"{action=!r} ({type(action)}) not found in action_space."
            raise InvalidOperationError(msg)
        if self.rng.rand() < self.p_switch:
            self.context = 1 - self.context

        correct_action = self.ob2action_context1[self.ob] if self.context == 0 else self.ob2action_context2[self.ob]

        reward = (correct_action == action) * 1.0

        # new observation
        self.ob = self.rng.randint(0, 2)
        obs = np.array([0.0, 0.0])
        obs[self.ob] = 1.0

        terminated = False
        truncated = False

        if not terminated:
            pass
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned terminated = True.\n",
                    "You should always call 'reset()' once you receive 'terminated = True'.",
                    "-- any further steps are undefined behavior.",
                )
            self.steps_beyond_done += 1

        return obs, reward, terminated, truncated, {}

    def reset(self):
        self.ob = 0
        self.context = 0
        self.steps_beyond_done = None
        obs = np.array([0.0, 0.0])
        obs[self.ob] = 1.0
        return obs, {}

    def render(self, mode="human") -> NoReturn:
        raise NotImplementedError

    def close(self) -> None:
        if self.viewer:
            self.viewer.close()


class FullInput(gym.Wrapper):
    """Lever pressing environment where reward and action is input."""

    def __init__(self, env) -> None:
        super().__init__(env)
        # Modify observation space to include reward and action
        orig_ob_space = self.observation_space
        # ob_shape is not used, so it is removed
        low = np.array(list(orig_ob_space.low) + [-1] + [0] * self.action_space.n)  # type: ignore[attr-defined]
        high = np.array(list(orig_ob_space.high) + [1] + [1] * self.action_space.n)  # type: ignore[attr-defined]
        self.observation_space: spaces.Box = spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # include reward and action information
        one_hot_action = [0.0] * self.action_space.n  # type: ignore[attr-defined]
        one_hot_action[action] = 1.0
        obs = np.array([*list(obs), reward, *one_hot_action])
        return obs, reward, terminated, truncated, info

    def reset(self):
        obs, _ = self.env.reset()
        # observation, reward, actions
        obs = np.array(list(obs) + [0.0] + [0.0] * self.action_space.n)
        return obs, {}
