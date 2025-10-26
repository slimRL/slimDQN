# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/dopamine/jax/replay_memory/replay_buffer.py
"""Simpler implementation of the standard DQN replay memory."""
from collections import OrderedDict, deque
from dataclasses import dataclass
import operator

import jax
import numpy as np

from flax import struct
import snappy

from slimdqn.sample_collection.samplers import Uniform, Prioritized


@dataclass
class TransitionElement:
    observation: np.ndarray[np.float64]
    action: np.uint
    reward: np.float32
    is_terminal: bool
    episode_end: bool


class ReplayElement(struct.PyTreeNode):
    """A single replay transition element supporting compression."""

    state: np.ndarray[np.float64]
    action: np.uint
    reward: np.float32
    next_state: np.ndarray[np.float64]
    is_terminal: bool

    @staticmethod
    def compress(buffer: np.ndarray) -> np.ndarray:
        compressed = np.frombuffer(snappy.compress(buffer), dtype=np.uint8)

        return np.array(
            (compressed, buffer.shape, buffer.dtype.str),
            dtype=[
                ("data", "u1", compressed.shape),
                ("shape", "i4", (len(buffer.shape),)),
                ("dtype", f"S{len(buffer.dtype.str)}"),
            ],
        )

    @staticmethod
    def uncompress(compressed: np.ndarray) -> np.ndarray:
        shape = tuple(compressed["shape"])
        dtype = compressed["dtype"].item()
        compressed_bytes = compressed["data"].tobytes()
        uncompressed = snappy.uncompress(compressed_bytes)
        return np.ndarray(shape=shape, dtype=dtype, buffer=uncompressed)

    def pack(self):
        return self.replace(
            state=ReplayElement.compress(self.state), next_state=ReplayElement.compress(self.next_state)
        )

    def unpack(self):
        return self.replace(
            state=ReplayElement.uncompress(self.state), next_state=ReplayElement.uncompress(self.next_state)
        )


class ReplayBuffer:
    def __init__(
        self,
        sampling_distribution: Uniform | Prioritized,
        max_capacity: int,
        batch_size: int,
        stack_size: int,
        update_horizon: int,
        gamma: float,
        clipping: callable,
    ):
        self.add_count = 0
        self.max_capacity = max_capacity
        # Only stores complete samples
        self.memory = OrderedDict[int, ReplayElement]()

        self.sampling_distribution = sampling_distribution
        self.batch_size = batch_size

        self.stack_size = stack_size
        self.update_horizon = update_horizon
        self.gamma = gamma
        self.clipping = clipping

        # Temporarily stores transitions before transfering them to the memory
        self.subtrajectory_tail = deque[TransitionElement](maxlen=self.update_horizon + self.stack_size)
        # Useful to know if subtrajectory_tail corresponds to the beginning of a trajectory
        self.beginning_trajectory = True

    def make_replay_element(self) -> ReplayElement:
        subtrajectory_len = len(self.subtrajectory_tail)

        # We construct a ReplayElement in the following scenarios:
        # (1) subtrajectory_tail is full size: this results in a non-terminating ReplayElement.
        # (2) subtrajectory_tail is not full size and the subtrajectory corresponds to beginning of trajectory.
        #   In this case, either the sample is terminal and a ReplayElement can be constructed, or it needs at least n+1 observations.
        # (3) subtrajectory_tail is not full size, subtrajectory_tail represents the end of the trajectory.
        #   In this case, a ReplayElement can be constructed only if the last observation leads to a terminal state
        #   and that there is enough observations to form a state (more than stack_size).
        if subtrajectory_len == self.update_horizon + self.stack_size:
            effective_horizon = self.update_horizon
            is_terminal = False
        elif subtrajectory_len > 0 and self.beginning_trajectory:
            if self.subtrajectory_tail[-1].is_terminal:
                if subtrajectory_len >= self.stack_size:
                    effective_horizon = subtrajectory_len - self.stack_size
                else:
                    effective_horizon = 0
                is_terminal = True
            elif subtrajectory_len >= self.update_horizon + 1:
                effective_horizon = subtrajectory_len - 1
                is_terminal = False
            else:
                return None
        elif subtrajectory_len >= self.stack_size and self.subtrajectory_tail[-1].is_terminal:
            effective_horizon = subtrajectory_len - self.stack_size
            is_terminal = True
        else:
            return None

        observation_shape = self.subtrajectory_tail[-1].observation.shape + (self.stack_size,)
        observation_dtype = self.subtrajectory_tail[-1].observation.dtype

        o_tm1 = np.zeros(observation_shape, observation_dtype)
        o_tm1_slice = slice(
            subtrajectory_len - effective_horizon - self.stack_size, subtrajectory_len - effective_horizon - 1
        )
        a_tm1 = self.subtrajectory_tail[o_tm1_slice.stop].action

        o_t = np.zeros(observation_shape, observation_dtype)
        o_t_slice = slice(subtrajectory_len - self.stack_size, subtrajectory_len - 1)

        gamma_slice = slice(o_tm1_slice.stop, o_tm1_slice.stop + self.update_horizon - 1)

        # We iterate through the n-step trajectory and compute the
        # cumulant and insert the observations into the appropriate stacks
        r_t = 0.0
        for t, transition_t in enumerate(self.subtrajectory_tail):
            # If we should be accumulating reward at index t for an n-step return?
            if gamma_slice.start <= t <= gamma_slice.stop:
                r_t += transition_t.reward * (self.gamma ** (t - gamma_slice.start))

            # If we should be accumulating frames for the frame-stack?
            if o_tm1_slice.start <= t <= o_tm1_slice.stop:
                o_tm1[..., t - o_tm1_slice.start] = transition_t.observation
            if o_t_slice.start <= t <= o_t_slice.stop:
                o_t[..., t - o_t_slice.start] = transition_t.observation

        return ReplayElement(state=o_tm1, action=a_tm1, reward=r_t, next_state=o_t, is_terminal=is_terminal)

    def accumulate(self, transition: TransitionElement):
        """Add a transition to the accumulator, maybe receive valid ReplayElements.

        If the transition has a terminal or end of episode signal, it will create a
        new trajectory and yield multiple elements.
        """
        if len(self.subtrajectory_tail) == self.update_horizon + self.stack_size:
            self.beginning_trajectory = False
        self.subtrajectory_tail.append(transition)

        if transition.is_terminal:
            while replay_element := self.make_replay_element():
                yield replay_element
                self.subtrajectory_tail.popleft()
            self.subtrajectory_tail.clear()
            self.beginning_trajectory = True
        else:
            if replay_element := self.make_replay_element():
                yield replay_element
            # If the transition truncates the trajectory then clear it
            if transition.episode_end:
                self.subtrajectory_tail.clear()
                self.beginning_trajectory = True

    def add(self, transition: TransitionElement):
        for replay_element in self.accumulate(transition):
            self.memory[self.add_count] = replay_element.pack()
            self.sampling_distribution.add(self.add_count)
            self.add_count += 1

            if self.add_count > self.max_capacity:
                oldest_key, _ = self.memory.popitem(last=False)
                self.sampling_distribution.remove(oldest_key)

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        sample_keys, importance_weights = self.sampling_distribution.sample(batch_size)
        replay_elements = operator.itemgetter(*sample_keys)(self.memory)
        replay_elements = map(operator.methodcaller("unpack"), replay_elements)
        return jax.tree_util.tree_map(lambda *xs: np.stack(xs), *replay_elements), importance_weights

    def update(self, keys, loss):
        # update function for Prioritized sampler
        self.sampling_distribution.update(keys, loss)
