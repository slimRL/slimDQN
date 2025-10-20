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
        if not buffer.flags["C_CONTIGUOUS"]:
            buffer = buffer.copy(order="C")
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
        stack_size: int = 4,
        update_horizon: int = 1,
        gamma: float = 0.99,
        clipping: callable = None,
    ):
        self.add_count = 0
        self.max_capacity = max_capacity
        self.memory = OrderedDict[int, ReplayElement]()

        self.sampling_distribution = sampling_distribution
        self.batch_size = batch_size

        self.stack_size = stack_size
        self.update_horizon = update_horizon
        self.gamma = gamma
        self.clipping = clipping

        self.subtrajectory_tail = deque[TransitionElement](maxlen=self.update_horizon + self.stack_size)

    def make_replay_element(self) -> ReplayElement:

        subtrajectory_len = len(self.subtrajectory_tail)
        last_transition = self.subtrajectory_tail[-1]

        # Check if we have a valid transition, i.e. we either
        #   1) have accumulated more transitions than the update horizon
        #   2) have a trajectory shorter than the update horizon, but the
        #      last element is terminal
        if not (subtrajectory_len > self.update_horizon or (subtrajectory_len > 1 and last_transition.is_terminal)):
            return None
        else:
            # Calculate effective horizon, this can differ from the update horizon
            # when we have n-step transitions where the last observation is terminal.
            if last_transition.is_terminal and subtrajectory_len <= self.update_horizon:
                effective_horizon = subtrajectory_len - 1
            else:
                effective_horizon = self.update_horizon

            observation_shape = last_transition.observation.shape + (self.stack_size,)
            observation_dtype = last_transition.observation.dtype

            o_tm1 = np.zeros(observation_shape, observation_dtype)
            # Initialize the slice for which this observation is valid.
            # The start index for o_tm1 is the start of the n-step trajectory.
            # The end index for o_tm1 is just moving over `stack size`.
            o_tm1_slice = slice(
                subtrajectory_len - effective_horizon - self.stack_size, subtrajectory_len - effective_horizon - 1
            )
            # The action chosen will be the last transition in the stack.
            a_tm1 = self.subtrajectory_tail[o_tm1_slice.stop].action

            o_t = np.zeros(observation_shape, observation_dtype)
            # Initialize the slice for which this observation is valid.
            # The start index for o_t is just moving backwards `stack size`.
            # The end index for o_t is just the last index of the n-step trajectory.
            o_t_slice = slice(subtrajectory_len - self.stack_size, subtrajectory_len - 1)
            # Terminal information will come from the last transition in the stack
            is_terminal = self.subtrajectory_tail[o_t_slice.stop].is_terminal

            # Slice to accumulate n-step returns. This will be the end
            # transition of o_tm1 plus the effective horizon.
            # This might over-run the trajectory length in the case of n-step
            # returns where the last transition is terminal.
            gamma_slice = slice(o_tm1_slice.stop, o_tm1_slice.stop + self.update_horizon - 1)

            # Now we'll iterate through the n-step trajectory and compute the
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
        self.subtrajectory_tail.append(transition)

        if transition.is_terminal:
            while replay_element := self.make_replay_element():
                yield replay_element
                self.subtrajectory_tail.popleft()
            self.subtrajectory_tail.clear()
        else:
            if replay_element := self.make_replay_element():
                yield replay_element
            # If the transition truncates the trajectory then clear it
            if transition.episode_end:
                self.subtrajectory_tail.clear()

    def add(self, transition: TransitionElement):
        for replay_element in self.accumulate(transition):
            replay_element = replay_element.pack()
            self.memory[self.add_count] = replay_element
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
