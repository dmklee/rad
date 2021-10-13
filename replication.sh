#!/bin/bash

#ENVS= (finger-spin cartpole-swingup reacher-easy cheetah-run walker-walk cup-catch)
ENVS=( reacher-easy )
# cheetah-run walker-walk cup-catch )
AGENTS=( rad_sac pixel_sac )
SEEDS=( 1 2 3 )

for seed in $SEEDS
do
	for env in $ENVS
	do
		for agent in $AGENTS
		do
			python -m scripts.replication --seeds $seed --envs $env --agent $agent || true
		done
	done
done
