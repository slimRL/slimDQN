launch_job/atari/local_dqn.sh --experiment_name test_Breakout  --first_seed 1 --last_seed 1 --architecture_type cnn \
    --features 32 64 64 512 --n_initial_samples 20_000 --replay_buffer_capacity 1_000_000 --n_epochs 200 -ntspe 250_000 \
    --epsilon_duration 250_000 --horizon 27_000 --target_update_period 8_000 --learning_rate 6.25e-4 --data_to_update 4 --disable_wandb