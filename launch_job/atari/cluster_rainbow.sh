#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

echo "launch train $ALGO_NAME"

if [[ $GPU = true ]]
then
    sbatch --job-name $EXPERIMENT_NAME-$ALGO_NAME --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=4 --mem-per-cpu=$((N_PARALLEL_SEEDS * 3))G --time=15:00:00 --gres=gpu:1 --prefer="rtx3090|a5000" --partition gpu \
    --output=experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/train_$((N_PARALLEL_SEEDS * (FIRST_SEED - 1) + 1))-$((N_PARALLEL_SEEDS * LAST_SEED)).out \
    launch_job/$ENV_NAME/train.sh --algo_name $ALGO_NAME --env_name $ENV_NAME --experiment_name $EXPERIMENT_NAME $ARGS --n_parallel_seeds $N_PARALLEL_SEEDS -gpu
else
    sbatch --job-name $EXPERIMENT_NAME-$ALGO_NAME --array=$FIRST_SEED-$LAST_SEED --cpus-per-task=4 --mem-per-cpu=$((N_PARALLEL_SEEDS * 3))G --time=15:00:00 --partition amd,amd2,amd3 \
    --output=experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/train_$((N_PARALLEL_SEEDS * (FIRST_SEED - 1) + 1))-$((N_PARALLEL_SEEDS * LAST_SEED)).out \
    launch_job/$ENV_NAME/train.sh --algo_name $ALGO_NAME --env_name $ENV_NAME --experiment_name $EXPERIMENT_NAME $ARGS --n_parallel_seeds $N_PARALLEL_SEEDS
fi
