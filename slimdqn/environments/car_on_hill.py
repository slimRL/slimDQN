# Reference: https://github.com/MushroomRL/mushroom-rl.git

import jax
import numpy as np
from scipy.integrate import odeint

CAR_ON_HILL_DEFAULT_HORIZON = 200


class CarOnHill:
    """
    The Car On Hill environment as presented in:
    "Tree-Based Batch Mode Reinforcement Learning". Ernst D. et al.. 2005.

    """

    def __init__(self):
        # MDP parameters
        self.max_pos = 1.0
        self.max_velocity = 3.0
        self._g = 9.81
        self._m = 1.0
        self._discrete_actions = [-4.0, 4.0]

        # MDP properties
        self._dt = 0.1

        # Visualization
        self.observation_shape = (2,)
        self.n_actions = 2

    @property
    def observation(self) -> np.ndarray:
        return np.copy(self.state_)

    @property
    def state(self) -> np.ndarray:
        return np.copy(self.state_)

    def reset(self, state=None, **kwargs):
        if state is None:
            self.state_ = np.array([-0.5, 0])
        else:
            self.state_ = state

        self.n_steps = 0

    def step(self, action):
        self.n_steps += 1
        action = self._discrete_actions[action]
        sa = np.append(self.state_, action)
        new_state = odeint(self._dpds, sa, [0, self._dt])

        self.state_ = new_state[-1, :-1]

        if self.state_[0] < -self.max_pos or np.abs(self.state_[1]) > self.max_velocity:
            reward = -1.0
            absorbing = True
        elif self.state_[0] > self.max_pos and np.abs(self.state_[1]) <= self.max_velocity:
            reward = 1.0
            absorbing = True
        else:
            reward = 0.0
            absorbing = False

        return reward, absorbing

    @staticmethod
    def _angle(x):
        if x < 0.5:
            m = 4 * x - 1
        else:
            m = 1 / ((20 * x**2 - 20 * x + 6) ** 1.5)

        return np.arctan(m)

    @staticmethod
    def _height(x):
        y_neg = 4 * x**2 - 2 * x
        y_pos = (2 * x - 1) / np.sqrt(5 * (2 * x - 1) ** 2 + 1)
        y = np.zeros_like(x)

        mask = x < 0.5
        neg_mask = np.logical_not(mask)
        y[mask] = y_neg[mask]
        y[neg_mask] = y_pos[neg_mask]

        y_norm = (y + 1) / 2

        return y_norm

    def _dpds(self, state_action, t):
        position = state_action[0]
        velocity = state_action[1]
        u = state_action[-1]

        if position < 0.0:
            diff_hill = 2 * position + 1
            diff_2_hill = 2
        else:
            diff_hill = 1 / ((1 + 5 * position**2) ** 1.5)
            diff_2_hill = (-15 * position) / ((1 + 5 * position**2) ** 2.5)

        dp = velocity
        ds = (u - self._g * self._m * diff_hill - velocity**2 * self._m * diff_hill * diff_2_hill) / (
            self._m * (1 + diff_hill**2)
        )

        return dp, ds, 0.0