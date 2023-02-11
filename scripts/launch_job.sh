#!/bin/bash

TASK="${TASK:-gym}" # d4rl / antmaze / rl_unplugged
PRIORITY="${PRIORITY:-low}"
ALGO="${ALGO:-DiffQL}"
RUNS="${RUNS:-1}" # d4rl / antmaze / rl_unplugged
DBMODE="${DBMODE:-mcmc}"
N_SAMPLES="${N_SAMPLES:-50}"
QF_LAYER_NORM="${QF_LAYER_NORM:-False}"
POLICY_LAYER_NORM="${POLICY_LAYER_NORM:-False}"
OBS_NORM="${OBS_NORM:-False}"
LOSS_TYPE="${LOSS_TYPE:-TD3}"
SAMPLE_METHOD="${SAMPLE_METHOD:-ddpm}"
WEIGHT_MODE="${WEIGHT_MODE:-mle}"
AVG_FN="${AVG_FN:-mean}"
CRR_FN="${CRR_FN:-exp}"
ADV_NORM="${ADV_NORM:-False}"
GUIDE_COEF="${GUIDE_COEF:-1.0}"
NORM_REW="${NORM_REW:-False}"
LR_DECAY="${LR_DECAY:-True}"
FIXED_STD="${FIXED_STD:-True}"

if [ "$SAMPLE_METHOD" = "ddpm" ];
then
  NUM_T=100
elif [ "$SAMPLE_METHOD" = 'dpm' ];
then
  NUM_T=1000
else
  echo "sample method not implemented"
fi

if [ "$LOSS_TYPE" = "IQL" ];
then
  if [ "TASK" = 'antmaze' ];
  then
    NORM_REW=False
  else
    NORM_REW=True
  fi
fi

if [ "$LOSS_TYPE" = "CRR" ];
then
  ADV_NORM=True
fi

BASE_CMD="python -m diffusion.trainer --logging.output_dir=./experiment_output --logging.online --algo=${ALGO} --obs_norm=${OBS_NORM} --algo_cfg.loss_type=${LOSS_TYPE} --sample_method=${SAMPLE_METHOD} --algo_cfg.crr_avg_fn=${AVG_FN} --algo_cfg.crr_fn=${CRR_FN} --algo_cfg.adv_norm=${ADV_NORM} --qf_layer_norm=${QF_LAYER_NORM} --policy_layer_norm=${POLICY_LAYER_NORM} --algo_cfg.num_timesteps=${NUM_T} --algo_cfg.guide_coef=${GUIDE_COEF} --norm_reward=${NORM_REW} --algo_cfg.lr_decay=${LR_DECAY} --algo_cfg.fixed_std=${FIXED_STD}"

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
    PRIORITY=${PRIORITY} NS=${NS} make run cmd="${BASE_CMD} --seed=${i}  --env=antmaze-${level} --eval_n_trajs=100 --eval_period=50 --n_epochs=2000 --algo_cfg.max_q_backup=True --algo_cfg.expectile=0.9 --algo_cfg.awr_temperature=10.0"
    sleep 1
  done
elif [ "$TASK" = "kitchen" ]; then
  for level in complete-v0 partial-v0 mixed-v0
  do
    PRIORITY=${PRIORITY} NS=${NS} make run cmd="${BASE_CMD} --seed=${i} --env=kitchen-${level} --n_epochs 1000 --algo_cfg.awr_temperature=0.5"
    sleep 1
  done
elif [ "$TASK" = "adroit" ]; then
  for scenario in pen
  do
    for tp in human cloned
    do
      PRIORITY=${PRIORITY} NS=${NS} make run cmd="${BASE_CMD} --algo=${ALGO} --seed=${i} --env=${scenario}-${tp}-v1 --n_epochs 1000 --algo_cfg.awr_temperature=0.5"
      sleep 1
    done
  done
else
  echo "wrong env name"
fi
done
