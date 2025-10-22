import os
import shutil
import subprocess
import unittest


class TestAtari(unittest.TestCase):
    def test_dqn(self):
        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "../experiments/atari/exp_output/_test_dqn_Pong"
        )
        if os.path.exists(save_path):
            shutil.rmtree(save_path)

        returncode = subprocess.run(
            [
                "python3",
                "experiments/atari/dqn.py",
                "--experiment_name",
                "_test_dqn_Pong",
                "--seed",
                "1",
                "--disable_wandb",
                "--features",
                "2",
                "3",
                "1",
                "15",
                "--replay_buffer_capacity",
                "100",
                "--batch_size",
                "3",
                "--update_horizon",
                "1",
                "--gamma",
                "0.99",
                "--learning_rate",
                "1e-4",
                "--horizon",
                "10",
                "--n_epochs",
                "1",
                "--n_training_steps_per_epoch",
                "10",
                "--data_to_update",
                "3",
                "--target_update_period",
                "3",
                "--n_initial_samples",
                "3",
                "--epsilon_end",
                "0.01",
                "--epsilon_duration",
                "4",
                "--architecture_type",
                "cnn",
            ]
        ).returncode
        assert returncode == 0, "The command should not have raised an error."

        shutil.rmtree(save_path)
