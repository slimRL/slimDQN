#!/bin/bash

function parse_arguments() {
    IFS='_' read -ra splitted_file_name <<< $(basename $0)
    ALGO_NAME=${splitted_file_name[-1]::-3}
    ENV_NAME=$(basename $(dirname ${0}))
    ARGS=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            --algo_name)
                ALGO_NAME=$2
                shift
                shift
                ;;
            --env_name)
                ENV_NAME=$2
                shift
                shift
                ;;
            -en | --experiment_name)
                EXPERIMENT_NAME=$2
                shift
                shift
                ;;
            -fs | --first_seed)
                FIRST_SEED=$2
                shift
                shift
                ;;
            -ls | --last_seed)
                LAST_SEED=$2
                shift
                shift
                ;;
            -nps | --n_parallel_seeds)
                N_PARALLEL_SEEDS=$2
                shift
                shift
                ;;
            -?* | ?*)
                ARGS="$ARGS $1"
                shift
                ;;
        esac
    done

    if [[ $EXPERIMENT_NAME == "" ]]
    then
        echo "experiment name is missing, use --experiment_name" >&2
        exit 1
    elif ( [[ $FIRST_SEED = "" ]] || [[ $LAST_SEED = "" ]] || [[ $FIRST_SEED -gt $LAST_SEED ]] )
    then
        echo "you need to specify --first_seed and --last_seed and make to sure that first_seed <= last_seed" >&2
        exit 1
    fi
    if [[ $N_PARALLEL_SEEDS == "" ]]
    then
        N_PARALLEL_SEEDS=1
    fi

    [ -d experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME ] || mkdir -p experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME

}