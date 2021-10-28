#!/bin/bash

ENVS=( reacher-easy cheetah-run finger-spin )
#AGENTS=( rad_sac pixel_sac )
SEEDS=( 0 1 2 )

CUDA_ID=$1
for env in "${ENVS[@]}"
do
	for seed in "${SEEDS[@]}"
	do
		tmp=$((seed + CUDA_ID * 10))
		#echo $CUDA_ID $agent $env $tmp
		CUDA_VISIBLE_DEVICES=$CUDA_ID python -m scripts.replication --seeds $tmp --envs $env --agent pixel_sac --encoder-type pixel_aa
	done
done
