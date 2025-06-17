SHARED_ARGS="--features 32 64 256 --replay_buffer_capacity 100_000 --batch_size 32 --update_horizon 20 --gamma 0.99 \
    --horizon 27_000 --n_epochs 20 --n_training_steps_per_epoch 5_000 --update_to_data 1 --n_initial_samples 1_600 \
    --epsilon_end 0.01 --epsilon_duration 2_000 --n_bins 50 --min_value -10 --max_value 10"

GAME="MsPacman"
ARCHITECTURE_TYPE="der"  # cnn impala der
TARGET_UPDATE_FREQ=2000

PLATFORM="local"  # cluster local

SHARED_ARGS="$SHARED_ARGS --target_update_frequency $TARGET_UPDATE_FREQ --architecture_type $ARCHITECTURE_TYPE"
SHARED_NAME="${ARCHITECTURE_TYPE}_T${TARGET_UPDATE_FREQ}"
# ----- C51 Loss -----
C51_ARGS="--learning_rate 1e-4"

RAINBOW_ARGS="--experiment_name atari100k_${SHARED_NAME}_${GAME}"
launch_job/atari/${PLATFORM}_rainbow.sh --first_seed 1 --last_seed 1 $SHARED_ARGS $C51_ARGS $RAINBOW_ARGS
