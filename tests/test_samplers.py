# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/tests/dopamine/jax/replay_memory/samplers_test.py
"""Testing samplers."""

from absl.testing import parameterized
from slimdqn.sample_collection import samplers
import numpy as np


class PrioritizedSamplingTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.sampler = samplers.Prioritized(seed=0, max_capacity=10)

    def test_sample(self):
        keys = [0, 1, 2, 3, 4]
        priorities = [1.0, 2.0, 3.0, 4.0, 0.0]

        for key, priority in zip(keys, priorities):
            self.sampler.add(key)
            self.sampler.update(key, priority)

        # test if zero priority absent
        samples = self.sampler.sample(5)
        np.testing.assert_array_less(samples, 4)

        self.sampler.update(keys=np.array([2, 3]), loss=np.array([0.0, 0.0]))

        # test if priority updated properly
        samples = self.sampler.sample(5)
        np.testing.assert_array_less(samples, 2)

        self.sampler.remove(0)
        samples = self.sampler.sample(5)
        np.testing.assert_array_almost_equal(samples, 1)
