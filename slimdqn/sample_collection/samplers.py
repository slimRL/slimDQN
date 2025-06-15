# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/dopamine/jax/replay_memory/samplers.py
"""Sampling distributions."""

import numpy as np
import numpy.typing as npt

import jax

from slimdqn.sample_collection import ReplayItemID
from slimdqn.sample_collection import sum_tree


class UniformSamplingDistribution:
    """A uniform sampling distribution."""

    def __init__(self, seed: int) -> None:
        self._rng_key = np.random.default_rng(seed)

        self._key_to_index = {}
        self._index_to_key = []

    def add(self, key: ReplayItemID) -> None:
        self._key_to_index[key] = len(self._index_to_key)
        self._index_to_key.append(key)

    def remove(self, key: ReplayItemID) -> None:
        assert key in self._key_to_index, ValueError(f"Key {key} not found.")

        index = self._key_to_index[key]

        # for efficient O(1) pop on the keys
        self._index_to_key[index], self._index_to_key[-1] = (
            self._index_to_key[-1],
            self._index_to_key[index],
        )
        self._key_to_index[self._index_to_key[index]] = index
        self._key_to_index.pop(self._index_to_key.pop())

    def sample(self, size: int):

        assert self._index_to_key, ValueError("No keys to sample from.")

        indices = self._rng_key.integers(len(self._index_to_key), size=size)

        return (
            np.fromiter(
                (self._index_to_key[index] for index in indices),
                dtype=np.int32,
                count=size,
            ),
            {},
        )


class PrioritizedSamplingDistribution(UniformSamplingDistribution):
    """A prioritized sampling distribution."""

    def __init__(
        self,
        seed: int,
        max_capacity: int,
        priority_exponent: float = 0.5,
    ) -> None:
        self._max_capacity = max_capacity
        self._priority_exponent = priority_exponent
        self._sum_tree = sum_tree.SumTree(self._max_capacity)
        super().__init__(seed=seed)

    def add(self, key: ReplayItemID) -> None:
        super().add(key)
        self._sum_tree.set(
            self._key_to_index[key],
            self._sum_tree.max_recorded_priority**self._priority_exponent,
        )

    def update(self, metadata) -> None:
        priorities = np.where(
            metadata["loss"] == 0.0, 0.0, metadata["loss"] ** self._priority_exponent
        )  # to handle negative priority_exponent
        self._sum_tree.set(
            np.fromiter((self._key_to_index[key] for key in metadata["keys"]), dtype=np.int32),
            priorities,
        )

    def remove(self, key: ReplayItemID) -> None:
        index = self._key_to_index[key]
        last_index = len(self._index_to_key) - 1
        if index == last_index:
            # If index and last_index are the same, simply set the priority to 0.0.
            self._sum_tree.set(index, 0.0)
        else:
            # Otherwise, swap priorities with current index and last index
            # as that's how we pop the key from our datastructure.
            # This will run in O(logn) where n is the # of elements in the tree
            self._sum_tree.set(
                np.asarray([index, last_index], dtype=np.int32),
                np.asarray([self._sum_tree.get(last_index), 0.0]),
            )
        super().remove(key)

    def sample(self, size: int):
        if self._sum_tree.root == 0.0:
            keys, _ = super().sample(size)
            return keys, {"probabilities": np.full_like(keys, 1.0 / len(self._index_to_key), dtype=np.float64)}

        targets = self._rng_key.uniform(0.0, self._sum_tree.root, size=size)
        indices = self._sum_tree.query(targets)
        return (
            np.fromiter(
                (self._index_to_key[index] for index in indices),
                count=size,
                dtype=np.int32,
            ),
            {"probabilities": self._sum_tree.get(indices) / self._sum_tree.root},
        )
