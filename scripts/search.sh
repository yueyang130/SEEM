#!/bin/bash

TASK="${TASK:-d4rl}" # d4rl / antmaze / rl_unplugged
PRIORITY="${PRIORITY:-low}"
ALGO="${ALGO:-MISA}"
RUNS="${RUNS:-1}" # d4rl / antmaze / rl_unplugged
DBMODE="${DBMODE:-mcmc}"
N_SAMPLES="${N_SAMPLES:-50}"
BASE_CMD="python -m experiments.main --logging.output_dir=./experiment_output --logging.online --algo_cfg.n_actions=${N_SAMPLES}"

for (( i=1; i<=${RUNS}; i++ ))
do
for level in umaze-v0 umaze-diverse-v0 medium-play-v0 medium-diverse-v0 large-play-v0 large-diverse-v0
do
  for bc_weight in 0.5 1.0 3.0
  do
    for add_pos in True
    do
      for ub_grad in True False
      do
        PRIORITY=${PRIORITY} NS=max make run cmd="${BASE_CMD} --seed=${i}  --env=antmaze-${level} --eval_n_trajs=100 --algo_cfg.bc_weight_misa=${bc_weight} --algo_cfg.lagrange=True --algo_cfg.add_positive=${add_pos} --use_layer_norm=True --algo_cfg.unbiased_grad=${ub_grad} --eval_period=50 --algo_cfg.use_automatic_entropy_tuning=True --algo_cfg.qf_lr=0.0001 --algo_cfg.target_action_gap=3.0 --obs_norm=False --n_epochs=2000"
  sleep 1
done
done
done
done
done

