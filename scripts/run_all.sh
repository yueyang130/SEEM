#!/bin/bash

ALGO="${ALGO:-ConservativeSAC}"
N_SAMPLES="${N_SAMPLES:-50}"

PRIORITY=high ALGO=${ALGO} N_SAMPLES=${N_SAMPLES} TASK=d4rl RUNS=5 bash ./scripts/launch_job.sh
PRIORITY=high ALGO=${ALGO} N_SAMPLES=${N_SAMPLES} TASK=kitchen RUNS=5 bash ./scripts/launch_job.sh
PRIORITY=high ALGO=${ALGO} N_SAMPLES=${N_SAMPLES} TASK=adroit RUNS=5 bash ./scripts/launch_job.sh
