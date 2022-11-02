#!/bin/bash

# Script to reproduce results
ns=offbench
prio=high

envs=( "halfcheetah-medium-v2" "hopper-medium-v2" "walker2d-medium-v2" \
	"halfcheetah-medium-expert-v2"  "hopper-medium-expert-v2" "walker2d-medium-expert-v2" \
	"halfcheetah-medium-replay-v2" "hopper-medium-replay-v2" "walker2d-medium-replay-v2" \
	)


# for ((i=0;i<1;i+=1)); do 
# 	for env in ${envs[*]}; do
# 		PRIORITY=$prio NS=$ns make run cmd="python main.py --env $env --seed $i \
# 			--critic_type v --td_type onestep --bc_eval_steps 1000000 \
# 			--reweight --tag bc_adv_reweight"
# 	    sleep 5;
# 	done
# done

# for ((i=0;i<1;i+=1)); do 
# 	for env in ${envs[*]}; do
# 		PRIORITY=$prio NS=$ns make run cmd="python main.py --env $env --seed $i \
# 			--critic_type vq --td_type onestep --bc_eval_steps 1000000 \
# 			--reweight --tag bc_adv_reweight"
# 	    sleep 5;
# 	done
# done


for ((i=0;i<1;i+=1)) ; do 
	for env in antmaze-umaze-v0 antmaze-umaze-diverse-v0 antmaze-medium-play-v0 \
		antmaze-medium-diverse-v0 antmaze-large-play-v0 antmaze-large-diverse-v0; do
			PRIORITY=$prio NS=$ns make run cmd="python main.py --env $env --seed $i --eval_episodes=100 --eval_freq=100000 \
			--critic_type v --td_type onestep --bc_eval_steps 1000000 \
			--reweight --tag bc_adv_reweight"
	    sleep 5;
	done
done

for ((i=0;i<1;i+=1)) ; do 
	for env in antmaze-umaze-v0 antmaze-umaze-diverse-v0 antmaze-medium-play-v0 \
		antmaze-medium-diverse-v0 antmaze-large-play-v0 antmaze-large-diverse-v0; do
			PRIORITY=$prio NS=$ns make run cmd="python main.py --env $env --seed $i --eval_episodes=100 --eval_freq=100000 \
			--critic_type vq --td_type onestep --bc_eval_steps 1000000 \
			--reweight --tag bc_adv_reweight"
	    sleep 5;
	done
done





# PRIORITY=low NS=offrl make run cmd="python main.py --env hopper-random-v2 --seed 0"
# PRIORITY=low NS=offrl make run cmd="python main.py --env antmaze-umaze-v0 --eval_episodes=100 --eval_freq=100000 --seed 0"