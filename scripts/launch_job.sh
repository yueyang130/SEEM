#!/bin/bash

TASK="${TASK:-gym}" # d4rl / antmaze / rl_unplugged
PRIORITY="${PRIORITY:-low}"
ALGO="${ALGO:-DiffusionQL}"
RUNS="${RUNS:-1}" # d4rl / antmaze / rl_unplugged
DBMODE="${DBMODE:-mcmc}"
N_SAMPLES="${N_SAMPLES:-50}"
USE_LAYER_NORM="${USE_LAYER_NORM:-False}"
OBS_NORM="${OBS_NORM:-False}"
LOSS_TYPE="${LOSS_TYPE:-TD3}"

BASE_CMD="python -m diffusion.trainer --logging.output_dir=./experiment_output --logging.online --algo=${ALGO} --algo_cfg.n_actions=${N_SAMPLES} --use_layer_norm=${USE_LAYER_NORM} --obs_norm=${OBS_NORM} --algo_cfg.loss_type=${LOSS_TYPE}"

for (( i=1; i<=${RUNS}; i++ ))
do
if [ "$TASK" = "gym" ];
then
  for env in halfcheetah hopper walker2d
  do
  for level in medium medium-replay medium-expert
  # for level in random
  do
    PRIORITY=${PRIORITY} NS=${NS} make run cmd="${BASE_CMD} --algo=${ALGO} --seed=${i} --env=${env}-${level}-v2"
    sleep 1
  done
  done
elif [ "$TASK" = "rl_unplugged" ]; then
  for env in finger_turn_hard humanoid_run cartpole_swingup cheetah_run fish_swim walker_stand walker_walk
  do
    PRIORITY=${PRIORITY} NS=${NS} make run cmd="${BASE_CMD} --algo=${ALGO} --dataset=rl_unplugged --seed=${i} --env=${env} --dataset rl_unplugged"
    sleep 1
  done
elif [ "$TASK" = "antmaze" ]; then
  for level in umaze-v0 umaze-diverse-v0 medium-play-v0 medium-diverse-v0 large-play-v0 large-diverse-v0
  do
    PRIORITY=${PRIORITY} NS=${NS} make run cmd="${BASE_CMD} --seed=${i}  --env=antmaze-${level} --eval_n_trajs=100 --algo_cfg.bc_weight_misa=0.5 --algo_cfg.lagrange=True --algo_cfg.add_positive=True --use_layer_norm=True --algo_cfg.unbiased_grad=True --eval_period=50 --algo_cfg.use_automatic_entropy_tuning=True --algo_cfg.qf_lr=0.0001 --algo_cfg.target_action_gap=3.0 --obs_norm=False --n_epochs=2000"
    sleep 1
  done
elif [ "$TASK" = "kitchen" ]; then
  for level in complete-v0 partial-v0 mixed-v0
  do
    PRIORITY=${PRIORITY} NS=${NS} make run cmd="${BASE_CMD} --seed=${i} --env=kitchen-${level} --n_epochs 1000 --algo_cfg.bc_weight_misa=3.0 --use_layer_norm=True --algo_cfg.detach_pi=True --algo_cfg.use_automatic_entropy_tuning=True --algo_cfg.unbiased_grad=True --obs_norm=False --algo_cfg.qf_lr=0.0001"
    sleep 1
  done
elif [ "$TASK" = "adroit" ]; then
  for scenario in pen hammer door relocate
  do
    for tp in human cloned
    do
      PRIORITY=${PRIORITY} NS=${NS} make run cmd="${BASE_CMD} --algo=${ALGO} --seed=${i} --env=${scenario}-${tp}-v0 --n_epochs 1000 --algo_cfg.use_automatic_entropy_tuning=False --algo_cfg.target_action_gap=3.0 --algo_cfg.bc_weight_misa=5.0 --algo_cfg.detach_pi=False --obs_norm=False --use_layer_norm=True"
      sleep 1
    done
  done
else
  echo "wrong env name"
fi
done
