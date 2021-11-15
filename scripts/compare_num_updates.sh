#!/bin/bash

#ENVS=( reacher-easy cheetah-run finger-spin )
ENVS=( cheetah-run )
NUM_UPDATES=( 1 2 4 )

NUM_TRAIN_STEPS=30000


for env in "${ENVS[@]}"
do
    for num_updates in "${NUM_UPDATES[@]}"
    do
        # RAD
        CUDA_VISIBLE_DEVICES=0 python -m scripts.replication --seeds 0 --envs $env \
            --agent rad_sac --num-updates $num_updates --num-train-steps $NUM_TRAIN_STEPS &
        CUDA_VISIBLE_DEVICES=1 python -m scripts.replication --seeds 1 --envs $env \
            --agent rad_sac --num-updates $num_updates --num-train-steps $NUM_TRAIN_STEPS &
        wait

        # PIXEL_SAC
        CUDA_VISIBLE_DEVICES=0 python -m scripts.replication --seeds 0 --envs $env \
            --agent pixel_sac --num-updates $num_updates --num-train-steps $NUM_TRAIN_STEPS &
        CUDA_VISIBLE_DEVICES=1 python -m scripts.replication --seeds 1 --envs $env \
            --agent pixel_sac --num-updates $num_updates --num-train-steps $NUM_TRAIN_STEPS &
        wait

        # f1(±2px)
        CUDA_VISIBLE_DEVICES=0 python -m scripts.replication --seeds 0 --envs $env --fmap-shifts 2::: \
            --agent pixel_sac --num-updates $num_updates --num-train-steps $NUM_TRAIN_STEPS &
        CUDA_VISIBLE_DEVICES=1 python -m scripts.replication --seeds 1 --envs $env --fmap-shifts 2::: \
            --agent pixel_sac --num-updates $num_updates --num-train-steps $NUM_TRAIN_STEPS &
        wait

        # f4(±2px)
        CUDA_VISIBLE_DEVICES=0 python -m scripts.replication --seeds 0 --envs $env --fmap-shifts :::2 \
            --agent pixel_sac --num-updates $num_updates --num-train-steps $NUM_TRAIN_STEPS &
        CUDA_VISIBLE_DEVICES=1 python -m scripts.replication --seeds 1 --envs $env --fmap-shifts :::2 \
            --agent pixel_sac --num-updates $num_updates --num-train-steps $NUM_TRAIN_STEPS &
        wait

    done
done
