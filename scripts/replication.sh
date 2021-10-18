#!/bin/bash

ENVS=( reacher-easy cheetah-run finger-spin )
#walker-walk cup-catch cartpole-swingup )
AGENTS=( rad_sac pixel_sac )
SEEDS=( 0 1 )

CUDA_ID=$1
for seed in "${SEEDS[@]}"
do
	for agent in "${AGENTS[@]}"
	do
		for env in "${ENVS[@]}"
		do
			tmp=$((seed + CUDA_ID * 10))
			#echo $CUDA_ID $agent $env $tmp
			CUDA_VISIBLE_DEVICES=$CUDA_ID python -m scripts.replication --seeds $tmp --envs $env --agent $agent > $agent$env$tmp.txt
		done
	done
done
