#!/bin/bash

export WANDB_API_KEY=3e0863e2d8f819730b85529bd24b3ebbb96d0eb3

TASK="${TASK:-gym}" # d4rl / antmaze / rl_unplugged
ALPHA="${ALPHA:-0.25}"
GUIDE_COEF="${GUIDE_COEF:-1.0}"
DIFF_COEF="${DIFF_COEF:-1.0}"

GPU="${GPU:-0}"
ALGO="${ALGO:-DiffQL}"
START="${START:-1}" # d4rl / antmaze / rl_unplugged
RUNS="${RUNS:-1}" # d4rl / antmaze / rl_unplugged
DBMODE="${DBMODE:-mcmc}"
N_SAMPLES="${N_SAMPLES:-50}"
QF_LAYER_NORM="${QF_LAYER_NORM:-False}"
ONLY_PENU_NORM="${ONLY_PENU_NORM:-False}"
POLICY_LAYER_NORM="${POLICY_LAYER_NORM:-False}"
OBS_NORM="${OBS_NORM:-False}"
LOSS_TYPE="${LOSS_TYPE:-Rainbow}"
USE_EXPECTILE="${USE_EXPECTILE:-False}"
EXPECTILE_Q="${EXPECTILE_Q:-False}"
AWR_TEMP="${AWR_TEMP:-10.0}"
SAMPLE_METHOD="${SAMPLE_METHOD:-dpm}"
WEIGHT_MODE="${WEIGHT_MODE:-mle}"
AVG_FN="${AVG_FN:-mean}"
CRR_FN="${CRR_FN:-exp}"
ADV_NORM="${ADV_NORM:-False}"
NORM_REW="${NORM_REW:-False}"
TARGET_CLIP="${TARGET_CLIP:-False}"
TRUST_REG="${TRUST_REG:-False}"
MAX_Q="${MAX_Q:-0}"
REW_SCALE="${REW_SCALE:-1}"
REW_BIAS="${REW_BIAS:-0}"
LR_DECAY="${LR_DECAY:-True}"
FIXED_STD="${FIXED_STD:-True}"
ORTHOG_INIT="${ORTHOG_INIT:-False}"
GUIDE_WARMUP="${GUIDE_WARMUP:-False}"
DIFF_ANNEAL="${DIFF_ANNEAL:-False}"
OPER="${OPER:-False}"
TWO_SAMPLER="${TWO_SAMPLER:-False}"
PRIORITY="${PRIORITY:-return}"
DIST_RL="${DIST_RL:-False}"
MAX_Q_BACKUP="${MAX_Q_BACKUP:-True}" # only for antmaze
LB_RATE="${LB_RATE:-1}" # only for antmaze
RESET_Q="${RESET_Q:-False}" # only for antmaze
RESET_ACTOR="${RESET_ACTOR:-False}" # only for antmaze
RESET_MODE="${RESET_MODE:-all}" # only for antmaze
RESET_INTERVAL="${RESET_INTERVAL:-500000}" # only for antmaze
MAX_TGT_Q="${MAX_TGT_Q:-False}" # only for antmaze
WEIGHT_DECAY="${WEIGHT_DECAY:-0}" # only for antmaze
EAS_TEMP="${EAS_TEMP:-1.0}"
STATE_SIGMA="${STATE_SIGMA:-0}"
ACTION_SIGMA="${ACTION_SIGMA:-0}"

if [ "$SAMPLE_METHOD" = "ddpm" ];
then
  NUM_T=100
elif [ "$SAMPLE_METHOD" = 'dpm' ];
then
  NUM_T=1000
else
  echo "sample method not implemented"
fi

if [ "$TASK" = 'antmaze' ];
then
    QF_LAYER_NORM=True
fi

# if [ "$LOSS_TYPE" = "IQL" ] || [  "$USE_EXPECTILE" = "True" ];
if [ "$LOSS_TYPE" = "IQL" ];
then
  # ORTHOG_INIT=True
  if [ "TASK" = 'antmaze' ];
  then
    NORM_REW=False
    FIXED_STD=True
    ADV_NORM=True
  else
    NORM_REW=True
    FIXED_STD=False
  fi
fi

BASE_CMD="WANDB_API_KEY=$WANDB_API_KEY python -m diffusion.trainer --logging.output_dir=./experiment_output --logging.online --logging.notes=$NOTES --algo=${ALGO} --obs_norm=${OBS_NORM} --algo_cfg.loss_type=${LOSS_TYPE} --algo_cfg.use_expectile=${USE_EXPECTILE}   --algo_cfg.expectile_q=${EXPECTILE_Q} --sample_method=${SAMPLE_METHOD} --algo_cfg.crr_avg_fn=${AVG_FN} --algo_cfg.crr_fn=${CRR_FN} --algo_cfg.adv_norm=${ADV_NORM} --qf_layer_norm=${QF_LAYER_NORM} --only_penultimate_norm=${ONLY_PENU_NORM} --policy_layer_norm=${POLICY_LAYER_NORM} --algo_cfg.num_timesteps=${NUM_T} --norm_reward=${NORM_REW} --reward_scale=${REW_SCALE} --reward_bias=${REW_BIAS} --algo_cfg.lr_decay=${LR_DECAY} --algo_cfg.fixed_std=${FIXED_STD} --orthogonal_init=${ORTHOG_INIT} \
--algo_cfg.crr_weight_mode=$WEIGHT_MODE --algo_cfg.guide_coef=$GUIDE_COEF --algo_cfg.trust_region_target=${TRUST_REG} --algo_cfg.target_clip=$TARGET_CLIP --algo_cfg.MAX_Q=$MAX_Q  --algo_cfg.diff_coef=$DIFF_COEF  --algo_cfg.alpha=$ALPHA --algo_cfg.guide_warmup=${GUIDE_WARMUP} --oper=$OPER --two_sampler=$TWO_SAMPLER --priority=$PRIORITY --algo_cfg.use_dist_rl=$DIST_RL --algo_cfg.diff_annealing=$DIFF_ANNEAL --lb_rate=$LB_RATE --algo_cfg.reset_q=$RESET_Q --algo_cfg.reset_actor=$RESET_ACTOR --algo_cfg.reset_mode=$RESET_MODE --algo_cfg.reset_interval=$RESET_INTERVAL --algo_cfg.max_tgt_q=$MAX_TGT_Q  --algo_cfg.weight_decay=$WEIGHT_DECAY --eas_temp=$EAS_TEMP --state_sigma=$STATE_SIGMA --action_sigma=$ACTION_SIGMA"



if [ "$DEBUG" = "True" ];
then
  PRIORITY=${PRIORITY} NS=${NS} make run cmd="${BASE_CMD} --algo=${ALGO} --seed=1 --env=halfcheetah-medium-v2 --n_epochs=20"
else
for (( i=$START; i<=${RUNS}; i++ ))
do
if [ "$TASK" = "gym" ];
then
  # for env in halfcheetah-medium-expert halfcheetah-medium
  # do
  #   echo "CUDA_VISIBLE_DEVICES=$GPU ${BASE_CMD} --algo=${ALGO} --seed=${i} --env=${env}-v2 --n_epochs=2000 &"
  #   sleep 1
  # done
  # for env in halfcheetah walker2d hopper
  for env in hopper
  do
  # for level in medium medium-replay medium-expert
  for level in medium medium-expert
  # for level in medium-replay
  do
    echo "CUDA_VISIBLE_DEVICES=$GPU ${BASE_CMD} --algo=${ALGO} --seed=${i} --env=${env}-${level}-v2 --n_epochs=2000 &"
  done
  done
elif [ "$TASK" = "rl_unplugged" ]; then
  for env in finger_turn_hard humanoid_run cartpole_swingup cheetah_run fish_swim walker_stand walker_walk
  do
    PRIORITY=${PRIORITY} NS=${NS} make run cmd="${BASE_CMD} --algo=${ALGO} --dataset=rl_unplugged --seed=${i} --env=${env} --dataset rl_unplugged &"
    sleep 1
  done
elif [ "$TASK" = "antmaze" ]; then
  # for level in umaze-v0 umaze-diverse-v0 medium-play-v0 medium-diverse-v0 large-play-v0 large-diverse-v0
  for level in large-play-v0 large-diverse-v0
  # for level in medium-play-v0 large-diverse-v0
  do
    # echo "CUDA_VISIBLE_DEVICES=$GPU ${BASE_CMD} --seed=${i}  --env=antmaze-${level} --eval_n_trajs=100 --eval_period=50 --n_epochs=2000 --algo_cfg.max_q_backup=$MAX_Q_BACKUP --algo_cfg.expectile=0.9 --algo_cfg.awr_temperature=$AWR_TEMP &"
    echo "CUDA_VISIBLE_DEVICES=$GPU ${BASE_CMD} --seed=${i}  --env=antmaze-${level} --eval_n_trajs=100 --eval_period=50 --algo_cfg.max_q_backup=$MAX_Q_BACKUP --algo_cfg.expectile=0.9 --algo_cfg.awr_temperature=$AWR_TEMP &"
    sleep 1
  done
elif [ "$TASK" = "kitchen" ]; then
  for level in complete-v0 partial-v0 mixed-v0
  do
    CUDA_VISIBLE_DEVICES=$GPU ${BASE_CMD} --seed=${i} --env=kitchen-${level} --n_epochs 1000 --algo_cfg.awr_temperature=0.5
    sleep 1
  done
elif [ "$TASK" = "adroit" ]; then
  for scenario in pen
  do
    for tp in human cloned
    do
      CUDA_VISIBLE_DEVICES=$GPU ${BASE_CMD} --algo=${ALGO} --seed=${i} --env=${scenario}-${tp}-v1 --n_epochs 1000 --algo_cfg.awr_temperature=0.5
      sleep 1
    done
  done
else
  echo "wrong env name"
fi
done
fi