import os
import sys

import jax
import numpy as np

from experiments.base.dqn import train
from experiments.base.utils import prepare_logs
from slimdqn.environments.atari import AtariEnv
from slimdqn.networks.dqn import DQN
from slimdqn.sample_collection.replay_buffer import ReplayBuffer
from slimdqn.sample_collection.samplers import UniformSamplingDistribution


def run(argvs=sys.argv[1:]):
    env_name, algo_name = os.path.abspath(__file__).split("/")[-2], os.path.abspath(__file__).split("/")[-1][:-3]
    p = prepare_logs(env_name, algo_name, argvs)

    q_key, train_key = jax.random.split(jax.random.PRNGKey(p["seed"]))

    env = AtariEnv(p["experiment_name"].split("_")[-1])
    rb = ReplayBuffer(
        sampling_distribution=UniformSamplingDistribution(p["seed"]),
        max_capacity=p["replay_buffer_capacity"],
        batch_size=p["batch_size"],
        update_horizon=p["update_horizon"],
        gamma=p["gamma"],
        clipping=lambda x: np.clip(x, -1, 1),
        stack_size=4,
        compress=True,
    )
    agent = DQN(
        q_key,
        (env.state_height, env.state_width, env.n_stacked_frames),
        env.n_actions,
        features=p["features"],
        architecture_type=p["architecture_type"],
        learning_rate=p["learning_rate"],
        gamma=p["gamma"],
        update_horizon=p["update_horizon"],
        data_to_update=p["data_to_update"],
        target_update_frequency=p["target_update_frequency"],
        adam_eps=1.5e-4,
    )
    train(train_key, p, agent, env, rb)


if __name__ == "__main__":
    run()
