# Inspired by dopamine implementation: https://github.com/google/dopamine/blob/master/tests/dopamine/jax/replaymemory/replay_buffer_test.py
from absl.testing import parameterized
import numpy as np

from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement, TransitionElement
from slimdqn.sample_collection.samplers import Uniform


# Default parameters used when creating the replay memory - mimic Atari.
OBSERVATION_SHAPE = (84, 84)
STACK_SIZE = 4


class ReplayBufferTest(parameterized.TestCase):

    def test_element_pack_unpack(self) -> None:
        """Pack and unpack a replay element."""
        state = np.zeros(OBSERVATION_SHAPE + (STACK_SIZE,), dtype=np.uint8)
        next_state = np.ones(OBSERVATION_SHAPE + (STACK_SIZE,), dtype=np.uint8)

        element = ReplayElement(state=state, action=0, reward=0, next_state=next_state, is_terminal=False)
        unpacked = element.pack().unpack()

        np.testing.assert_array_equal(unpacked.state, state)
        np.testing.assert_array_equal(unpacked.next_state, next_state)
        assert unpacked.action == 0
        assert unpacked.reward == 0
        assert unpacked.is_terminal == False

    def testAddUpToCapacity(self):
        rb = ReplayBuffer(
            sampling_distribution=Uniform(seed=0),
            max_capacity=10,
            batch_size=32,
            stack_size=STACK_SIZE,
            update_horizon=1,
            gamma=1.0,
            clipping=None,
        )

        transitions = []
        for i in range(16):
            transitions.append(TransitionElement(np.full(OBSERVATION_SHAPE, i), i, i, False, False))
            rb.add(transitions[-1])
        # Since we created the ReplayBuffer with a capacity of 10, it should have
        # gotten rid of the first 5 elements added.
        self.assertLen(rb.memory, 10)
        self.assertEqual(list(rb.memory.keys()), list(range(5, 15)))
        for i in range(5, 15):
            np.testing.assert_array_equal(
                ReplayElement.uncompress(rb.memory[i].state),
                np.array([transition.observation for transition in transitions[i - STACK_SIZE + 1 : i + 1]]).transpose(
                    1, 2, 0
                ),
            )
            np.testing.assert_array_equal(
                ReplayElement.uncompress(rb.memory[i].next_state),
                np.array([transition.observation for transition in transitions[i - STACK_SIZE + 2 : i + 2]]).transpose(
                    1, 2, 0
                ),
            )
            self.assertEqual(rb.memory[i].action, transitions[i].action)
            self.assertEqual(rb.memory[i].reward, transitions[i].reward)
            self.assertEqual(rb.memory[i].is_terminal, int(transitions[i].is_terminal))

    def testNSteprewards(self):
        rb = ReplayBuffer(
            sampling_distribution=Uniform(seed=0),
            max_capacity=10,
            batch_size=32,
            stack_size=STACK_SIZE,
            update_horizon=5,
            gamma=1.0,
            clipping=None,
        )

        for i in range(50):
            # add non-terminating observations with reward 2
            rb.add(TransitionElement(np.full(OBSERVATION_SHAPE, i), 0, 2.0, False, False))

        batch, _ = rb.sample()
        # Make sure the total reward is reward per step x update_horizon.
        np.testing.assert_array_equal(batch.reward, np.ones(32) * 10.0)

    def testSampleTransitionBatch(self):
        rb = ReplayBuffer(
            sampling_distribution=Uniform(seed=0),
            max_capacity=10,
            batch_size=32,
            stack_size=1,
            update_horizon=1,
            gamma=0.99,
            clipping=None,
        )

        for i in range(1, 21):
            terminal = i % 4 == 0  # Every 4th transition is terminal.
            rb.add(TransitionElement(np.full(OBSERVATION_SHAPE, i), 0, 0, terminal, False))

        # Verify we sample the expected indices by using the same rng state.
        self.rng = np.random.default_rng(seed=0)
        indices = self.rng.integers(len(rb.sampling_distribution.index_to_key), size=32)

        expected_states = [
            ReplayElement.uncompress(rb.memory[rb.sampling_distribution.index_to_key[i]].state) for i in indices
        ]
        expected_next_states = [
            ReplayElement.uncompress(rb.memory[rb.sampling_distribution.index_to_key[i]].next_state) for i in indices
        ]
        expected_terminal = [rb.memory[rb.sampling_distribution.index_to_key[i]].is_terminal for i in indices]

        batch, _ = rb.sample()
        np.testing.assert_array_equal(batch.state, expected_states)
        np.testing.assert_array_equal(batch.action, np.zeros(len(indices)))
        np.testing.assert_array_equal(batch.reward, np.zeros(len(indices)))
        np.testing.assert_array_equal(batch.next_state, expected_next_states)
        np.testing.assert_array_equal(batch.is_terminal, expected_terminal)
