#!/bin/bash

BASE_CMD="python -m experiments.main --logging.output_dir=./experiment_output --logging.online --eval_n_trajs 10"
TASK="${TASK:-d4rl}" # d4rl / antmaze / rl_unplugged
PRIORITY="${PRIORITY:-high}"
ALGO="${ALGO:-ConservativeSAC}"

for run in 1 2 3 4 5
do
for bc_weight in 0.1 0.3 0.5 0.8 1
do
for env in halfcheetah hopper walker2d
  do
  for level in medium medium-expert medium-replay
  do
    PRIORITY=${PRIORITY} NS=offrl make run cmd="${BASE_CMD} --algo=${ALGO} --env=${env}-${level}-v2 --bc_weight_ibal ${bc_weight}"
    sleep 1
  done
  done
done
done
