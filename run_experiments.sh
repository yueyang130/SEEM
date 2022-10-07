#!/bin/bash

# Script to reproduce results

envs=(
	# "halfcheetah-random-v2"
	# "hopper-random-v2"
	# "walker2d-random-v2"
	"halfcheetah-medium-v2"
	"hopper-medium-v2"
	"walker2d-medium-v2"
	# "halfcheetah-expert-v2"
	# "hopper-expert-v2"
	# "walker2d-expert-v2"
	"halfcheetah-medium-expert-v2"
	"hopper-medium-expert-v2"
	"walker2d-medium-expert-v2"
	"halfcheetah-medium-replay-v2"
	"hopper-medium-replay-v2"
	"walker2d-medium-replay-v2"
	)

# for ((i=0;i<5;i+=1))
# do 
# 	for env in ${envs[*]}
# 	do
# 		python main.py \
# 		--env $env \
# 		--seed $i
# 	done
# done

# python main.py --env halfcheetah-random-v2 --seed 0

for ((i=0;i<5;i+=1))
do 
	for env in ${envs[*]}
	do
		PRIORITY=low NS=offrl make run cmd="python main.py --env $env --seed $i \
			--reweight --reweight_eval=0 --reweight_improve=0  --reweight_constraint=1 --clip_constraint=1  --tag clip-v0"
	    sleep 5;
		
	done
done

for ((i=0;i<5;i+=1))
do 
	for env in ${envs[*]}
	do
		PRIORITY=low NS=offrl make run cmd="python main.py --env $env --seed $i \
			--reweight --reweight_eval=0 --reweight_improve=0  --reweight_constraint=1 --clip_constraint=2  --tag clip-v0"
	    sleep 5;
		
	done
done



# PRIORITY=low NS=offrl make run cmd="python main.py --env hopper-random-v2 --seed 0"