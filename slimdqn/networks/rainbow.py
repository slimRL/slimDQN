from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.core import FrozenDict

from slimdqn.networks.architectures.dqn import DQNNet
from slimdqn.sample_collection.replay_buffer import ReplayBuffer, ReplayElement


class Rainbow:
    def __init__(
        self,
        key: jax.random.PRNGKey,
        observation_dim,
        n_actions,
        n_bins: int,
        features: list,
        architecture_type: str,
        learning_rate: float,
        gamma: float,
        update_horizon: int,
        update_to_data: int,
        target_update_frequency: int,
        min_value: float,
        max_value: float,
        adam_eps: float = 1e-8,
    ):
        self.n_bins = n_bins
        self.network = DQNNet(features, architecture_type, n_actions * self.n_bins)
        self.network.apply_fn = lambda params, state: self.network.apply(params, state).reshape(
            (n_actions, self.n_bins)
        )
        self.params = self.network.init(key, jnp.zeros(observation_dim, dtype=jnp.float32))

        self.optimizer = optax.adam(learning_rate, eps=adam_eps)
        self.optimizer_state = self.optimizer.init(self.params)
        self.target_params = self.params

        self.gamma = gamma
        self.update_horizon = update_horizon
        self.update_to_data = update_to_data
        self.target_update_frequency = target_update_frequency
        self.cumulated_loss = 0
        self.cumulated_unsupported_prob = 0
        self.support = jnp.linspace(min_value, max_value, self.n_bins, dtype=jnp.float32)

    def update_online_params(self, step: int, replay_buffer: ReplayBuffer):
        if step % self.update_to_data == 0:
            batch_samples, metadata = replay_buffer.sample()

            self.params, self.optimizer_state, losses, unsupported_prob = self.learn_on_batch(
                self.params, self.target_params, self.optimizer_state, batch_samples, metadata["probabilities"]
            )
            metadata.update({"loss": losses})
            replay_buffer.update(metadata)

            self.cumulated_loss += losses.mean()
            self.cumulated_unsupported_prob += unsupported_prob

    def update_target_params(self, step: int):
        if step % self.target_update_frequency == 0:
            self.target_params = self.params.copy()

            logs = {
                "loss": self.cumulated_loss / (self.target_update_frequency / self.update_to_data),
                "unsupported_prob": self.cumulated_unsupported_prob
                / (self.target_update_frequency / self.update_to_data),
            }
            self.cumulated_loss = 0
            self.cumulated_unsupported_prob = 0

            return True, logs
        return False, {}

    @partial(jax.jit, static_argnames="self")
    def learn_on_batch(
        self, params: FrozenDict, params_target: FrozenDict, optimizer_state, batch_samples, batch_probabilities
    ):
        grad_loss, (losses, unsupported_prob) = jax.grad(self.loss_on_batch, has_aux=True)(
            params, params_target, batch_samples, batch_probabilities
        )
        updates, optimizer_state = self.optimizer.update(grad_loss, optimizer_state)
        params = optax.apply_updates(params, updates)

        return params, optimizer_state, losses, unsupported_prob

    def loss_on_batch(self, params: FrozenDict, params_target: FrozenDict, samples, batch_probabilities):
        losses, unsupported_probs = jax.vmap(self.loss, in_axes=(None, None, 0))(params, params_target, samples)
        loss_weights = 1.0 / jnp.sqrt(batch_probabilities + 1e-10)  # sqrt because beta is fixed to 0.5 in PER
        loss_weights /= jnp.max(loss_weights)
        return (losses * loss_weights).mean(), (losses, unsupported_probs.mean())

    def loss(self, params: FrozenDict, params_target: FrozenDict, sample: ReplayElement):
        # computes the loss for a single sample
        target_support, target_prob = self.compute_target(params_target, sample)
        q_logits = self.network.apply_fn(params, sample.state)[sample.action]
        projected_target, unsupported_prob = self.project_target_on_support(target_support, target_prob)
        cross_entropy = optax.softmax_cross_entropy(q_logits, projected_target)
        return cross_entropy, unsupported_prob

    def compute_target(self, params: FrozenDict, sample: ReplayElement):
        # computes the target value for single sample
        target_support = sample.reward + (1 - sample.is_terminal) * (self.gamma**self.update_horizon) * self.support
        target_logits = self.network.apply_fn(params, sample.next_state)
        target_prob = jax.nn.softmax(target_logits[jnp.argmax(jax.nn.softmax(target_logits) @ self.support)])
        return target_support, target_prob

    def project_target_on_support(self, target_support: jax.Array, target_prob: jax.Array) -> jax.Array:
        delta_z = (self.support[-1] - self.support[0]) / (self.n_bins - 1)
        clipped_support = jnp.clip(target_support, self.support[0], self.support[-1])
        return (
            jnp.clip(1 - jnp.abs(clipped_support - self.support[:, None]) / delta_z, 0, 1) @ target_prob,
            (
                (
                    jnp.clip(1 - jnp.abs(clipped_support - self.support[:, None]) / delta_z, 0, 1)[jnp.array([0, -1])]
                    == 1  # just take probabilities beyond vmin (0) and vmax (-1)
                )
                @ target_prob
            ).sum(),
        )

    @partial(jax.jit, static_argnames="self")
    def best_action(self, params: FrozenDict, state: jnp.ndarray, **kwargs):
        # computes the best action for a single state
        # We first compute the probabilities by applying the softmax on the last axis (bin axis).
        # Then, we compute the expectation by multiplying with the bin centers.
        return jnp.argmax(jax.nn.softmax(self.network.apply_fn(params, state)) @ self.support)

    def get_model(self):
        return {"params": self.params}
