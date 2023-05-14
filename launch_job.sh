#!/bin/bash

export WANDB_API_KEY=3e0863e2d8f819730b85529bd24b3ebbb96d0eb3

TASK="${TASK:-gym}" # d4rl / antmaze / rl_unplugged
GPU="${GPU:-0}"
START="${START:-1}" 
RUNS="${RUNS:-1}" 
ALPHA="${ALPHA:-2.5}"
BC_COEF="${BC_COEF:-1.0}"
QF_LAYER_NORM="${QF_LAYER_NORM:-0}"
REW_SCALE="${REW_SCALE:-1}"
REW_BIAS="${REW_BIAS:-0}"
ONLINE_PER="${ONLINE_PER:-1}"
PER_TEMP="${PER_TEMP:-0.6}"
RESAMPLE="${RESAMPLE:-False}"
TAU="${TAU:-0.005}"
OPTIMIZER="${OPTIMIZER:-adam}"
DISCOUNT="${DISCOUNT:-0.99}"

# BASE_CMD="WANDB_API_KEY=$WANDB_API_KEY python main.py --bc_eval=0 --alpha=$ALPHA --bc_coef=$BC_COEF --qf_layer_norm=$QF_LAYER_NORM --reward_scale=$REW_SCALE \
# --reward_bias=$REW_BIAS --online_per=$ONLINE_PER --per_temp=$PER_TEMP --tag=$TAG"
BASE_CMD="WANDB_API_KEY=$WANDB_API_KEY python main.py --bc_eval=0 --alpha=$ALPHA --bc_coef=$BC_COEF --qf_layer_norm=$QF_LAYER_NORM --reward_scale=$REW_SCALE \
--reward_bias=$REW_BIAS --tau=$TAU --tag=$TAG --optimizer=$OPTIMIZER --discount=$DISCOUNT"

if [ "$RESAMPLE" = "True" ];then
  BASE_CMD="$BASE_CMD --resample"
fi

for (( i=$START; i<=${RUNS}; i++ ))
do
  if [ "$TASK" = "gym" ];
  then
    # for env in halfcheetah-medium-expert halfcheetah-medium
    # do
    #   echo "CUDA_VISIBLE_DEVICES=$GPU ${BASE_CMD} --seed=${i} --env=${env} &"
    #   sleep 1
    # done
    # for env in hopper halfcheetah walker2d
    for env in hopper
    # for env in 
    do
    # for level in medium medium-replay medium-expert
    # for level in medium medium-expert
    for level in medium-replay
    do
      echo "CUDA_VISIBLE_DEVICES=$GPU ${BASE_CMD} --seed=${i} --env=${env}-${level}-v2 --max_timesteps=400000 &"
    done
    done
  elif [ "$TASK" = "rl_unplugged" ]; then
    for env in finger_turn_hard humanoid_run cartpole_swingup cheetah_run fish_swim walker_stand walker_walk
    do
      sleep 1
    done
  elif [ "$TASK" = "antmaze" ]; then
    for level in umaze-v0 umaze-diverse-v0 medium-play-v0 medium-diverse-v0 large-play-v0 large-diverse-v0
    # for env in umaze-diverse-v0 large-diverse-v0
    # for env in medium-play-v0 large-diverse-v0
    do
      echo "CUDA_VISIBLE_DEVICES=$GPU ${BASE_CMD} --seed=${i} --env=antmaze-${level} --eval_freq=50000 --eval_episodes=100 &"
      sleep 1
    done
  elif [ "$TASK" = "kitchen" ]; then
    for level in complete-v0 partial-v0 mixed-v0
    do
      sleep 1
    done
  elif [ "$TASK" = "adroit" ]; then
    for scenario in pen
    do
      for tp in human cloned
      do
        sleep 1
      done
    done
  else
    echo "wrong env name"
  fi
done
