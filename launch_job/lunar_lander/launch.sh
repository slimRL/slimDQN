#!/bin/bash

#policy expertise
EXPERTISE="noob"
FEATURES="5 5"
BATCH_SIZE=6

ENV_NAME="lunar_lander"
ALGO="dqn"

EXPERIMENT_TAG="${EXPERTISE}_${ALGO}_${ENV_NAME}"


FIRST_SEED=0
LAST_SEED=5
N_PARALLEL_SEEDS=1


BASE_ARGS="--architecture_type fc \
    --replay_buffer_capacity 10000 \
    --update_horizon 1 \
    --horizon 250 \
    --gamma 0.99 \
    --learning_rate 3e-3 \
    --n_epochs 30 \
    --target_update_period 200 \
    --n_states_1 17 \
    --n_states_2 17 \
    "

####################################################################################
POLICY_EXPERTISE_ARGS="--features $FEATURES --batch_size $BATCH_SIZE"


LAUNCHER_SCRIPT="launch_job/${ENV_NAME}/cluster_${ALGO}.sh"
FULL_EXPERIMENT_NAME="${EXPERIMENT_TAG}"

echo "----------------------------------------------------------------"
echo "Launching Job"
echo "Experiment: $FULL_EXPERIMENT_NAME"
echo "Algorithm:  $ALGO"
echo "Policy Expertise Args: $ALGO_ARGS"
echo "Data Args: $DATA_ARGS"
echo "----------------------------------------------------------------"

bash $LAUNCHER_SCRIPT \
    --experiment_name $FULL_EXPERIMENT_NAME \
    --first_seed $FIRST_SEED \
    --last_seed $LAST_SEED \
    --n_parallel_seeds $N_PARALLEL_SEEDS \
    $BASE_ARGS $POLICY_EXPERTISE_ARGS $DATA_ARGS