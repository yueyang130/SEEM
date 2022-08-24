#!/bin/bash

ALGO="${ALGO:-ConservativeSAC}"

PRIORITY=high ALGO=${ALGO} TASK=d4rl RUNS=5 bash ./scripts/launch_job.sh
PRIORITY=high ALGO=${ALGO} TASK=kitchen RUNS=5 bash ./scripts/launch_job.sh
PRIORITY=high ALGO=${ALGO} TASK=adroit RUNS=5 bash ./scripts/launch_job.sh