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
SAMPLE_METHOD="${SAMPLE_METHOD:-dpm}"
WEIGHT_MODE="${WEIGHT_MODE:-mle}"
AVG_FN="${AVG_FN:-mean}"
CRR_FN="${CRR_FN:-exp}"
ADV_NORM="${ADV_NORM:-False}"
GUIDE_COEF="${GIODE_COEF:-0.01}"

BASE_CMD="python -m diffusion.trainer --logging.output_dir=./experiment_output --logging.online --algo=${ALGO} --use_layer_norm=${USE_LAYER_NORM} --obs_norm=${OBS_NORM} --algo_cfg.loss_type=${LOSS_TYPE} --sample_method=${SAMPLE_METHOD} --algo_cfg.crr_avg_fn=${AVG_FN} --algo_cfg.crr_fn=${CRR_FN} --algo_cfg.crr_adv_norm=${ADV_NORM} --algo_cfg.guide_coef=${GUIDE_COEF}"

for env in hopper-medium-expert walker2d-medium-replay
do
  PRIORITY=${PRIORITY} NS=${NS} make run cmd="${BASE_CMD} --algo=${ALGO} --env=${env}-v2"
  sleep 1
done

