# slimDQN - simple, minimal and flexible implementation of Deep Q-Network

![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)
![jax_badge][jax_badge_link]
![Static Badge](https://img.shields.io/badge/lines%20of%20code-3060-green)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**`slimRL`** provides a concise and customizable implementation of Deep Q-Network (DQN) algorithm in Reinforcement Learning⛳ for Lunar Lander and Atari environments. 
It enables to quickly code and run proof-of-concept type of experiments in off-policy Deep RL settings.

### 🚀 Key advantages
✅ Easy to read - clears the clutter with minimal lines of code 🧹\
✅ Easy to experiment - flexible to play with algorithms and environments 📊\
✅ Fast to run - jax accleration, support for GPU and multiprocessing ⚡

<p align="center">
  <img width=48% src="images/lunar_lander.gif">
</p>


Let's dive in!

## User installation
CPU installation for Lunar Lander:
```bash
python3 -m venv env_cpu
source env_cpu/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .[dev]
```
GPU installation for Atari:
```bash
python3 -m venv env
source env/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .[dev,gpu]
```
To verify the installation, run the tests as:```pytest```

## Running experiments
To train a DQN agent on Lunar Lander on your local system, run:\
`
launch_job/lunar_lander/local_dqn.sh --experiment_name test_run --first_seed 0 --last_seed 0 --disable_wandb
`

It trains a DQN agent with 2 hidden layers of size 100, for 100_000 gradient steps. 

- To see the stage of training, you can check the logs in `experiments/lunar_lander/logs/test_run/dqn` folder
- The models and episodic returns are stored in `experiments/lunar_lander/exp_output/test_run/dqn` folder

To train on cluster:\
`
launch_job/lunar_lander/cluster_dqn.sh --experiment_name test_run --first_seed 0 --last_seed 0 --disable_wandb
`


[jax_badge_link]: https://tinyurl.com/5n8m53cy