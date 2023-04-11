# iql+td3
NOTES=diff-iql TASK=antmaze ALPHA=0 GUIDE_COEF=1 bash scripts/launch_job.sh # stable diff-iql with crr-v
NOTES=diff-iql TASK=antmaze ALPHA=0.25 GUIDE_COEF=1 bash scripts/launch_job.sh

NOTES=diff-iql+td3 TASK=gym ALPHA=0.25 GUIDE_COEF=1.0 bash scripts/launch_job.sh
NOTES=diff-iql+td3 TASK=gym ALPHA=0.25 GUIDE_COEF=0 bash scripts/launch_job.sh

NOTES=diff-iql+td3 TASK=gym ALPHA=0.25 GUIDE_COEF=0 EXPECTILE=0.5 bash scripts/launch_job.sh
NOTES=diff-iql+td3 TASK=gym ALPHA=0.25 GUIDE_COEF=0 EXPECTILE=0.6 bash scripts/launch_job.sh
NOTES=diff-iql+td3 TASK=gym ALPHA=0.25 GUIDE_COEF=0 EXPECTILE=0.9 bash scripts/launch_job.sh
NOTES=diff-iql+td3 TASK=gym ALPHA=0.25 GUIDE_COEF=0 EXPECTILE=0.95 bash scripts/launch_job.sh

NOTES=iql+maxq TASK=gym ALPHA=2.0 GUIDE_COEF=0 NORM_REW=True FIXED_STD=False bash scripts/launch_job.sh # iql+maxq
NOTES=iql+maxq TASK=gym ALPHA=2.0 GUIDE_COEF=0 QF_LAYER_NORM=True bash scripts/launch_job.sh # iql+maxq


# stable iql
ALGO=DiffIQL LOSS_TYPE=IQL AWR_TEMP=1.0 TASK=antmaze bash scripts/launch_job.sh
ALGO=DiffIQL LOSS_TYPE=IQL AWR_TEMP=0.5 TASK=antmaze bash scripts/launch_job.sh
ALGO=DiffIQL LOSS_TYPE=IQL AWR_TEMP=2.0 TASK=antmaze bash scripts/launch_job.sh
ALGO=DiffIQL LOSS_TYPE=IQL AWR_TEMP=4.0 TASK=antmaze bash scripts/launch_job.sh

ALGO=DiffIQL-ablate LOSS_TYPE=IQL TASK=gym QF_LAYER_NORM=False NORM_REW=True FIXED_STD=False bash scripts/launch_job.sh
ALGO=DiffIQL LOSS_TYPE=IQL TASK=gym QF_LAYER_NORM=True NORM_REW=False FIXED_STD=True bash scripts/launch_job.sh
ALGO=DiffIQL-ablate LOSS_TYPE=IQL TASK=gym QF_LAYER_NORM=True NORM_REW=False FIXED_STD=True ADV_NORM=True bash scripts/launch_job.sh

# quantile td3
ALGO=quantile_td3 TASK=gym ALPHA=2.0 GUIDE_COEF=0 bash scripts/launch_job.sh # reproduce TD3
ALGO=quantile_td3 TASK=gym ALPHA=0 GUIDE_COEF=1.0 bash scripts/launch_job.sh # quantile-v crr

ALGO=quantile_td3 TASK=antmaze EXPECTILE_Q=False ALPHA=2.0 GUIDE_COEF=0 QF_LAYER_NORM=True bash scripts/launch_job.sh # TD3
ALGO=quantile_td3 TASK=antmaze EXPECTILE_Q=False ALPHA=2.0 GUIDE_COEF=0 GUIDE_WARMUP=True QF_LAYER_NORM=True bash scripts/launch_job.sh # TD3 warmup guide
ALGO=quantile_td3 NOTES=update-qf TASK=antmaze EXPECTILE_Q=False ALPHA=2.0 GUIDE_COEF=0 GUIDE_WARMUP=True QF_LAYER_NORM=True bash scripts/launch_job.sh # TD3 warmup guide

ALGO=quantile_td3 TASK=antmaze EXPECTILE_Q=False ALPHA=1.0 GUIDE_COEF=0 QF_LAYER_NORM=True bash scripts/launch_job.sh # TD3 stable maze-diverse
ALGO=quantile_td3 TASK=antmaze EXPECTILE_Q=False ALPHA=0.5 GUIDE_COEF=0 QF_LAYER_NORM=True bash scripts/launch_job.sh # TD3 stable maze-diverse
ALGO=quantile_td3 TASK=antmaze EXPECTILE_Q=False ALPHA=0.2 GUIDE_COEF=0 QF_LAYER_NORM=True bash scripts/launch_job.sh # TD3 stable maze-diverse
ALGO=quantile_td3 TASK=antmaze EXPECTILE_Q=False ALPHA=0.1 GUIDE_COEF=0 QF_LAYER_NORM=True bash scripts/launch_job.sh # TD3 stable maze-diverse

# ALGO=quantile_td3 TASK=kitchen EXPECTILE_Q=False ALPHA=2.0 GUIDE_COEF=0 QF_LAYER_NORM=True bash scripts/launch_job.sh # TD3
# ALGO=quantile_td3 TASK=kitchen EXPECTILE_Q=False ALPHA=0.2 GUIDE_COEF=0 QF_LAYER_NORM=True bash scripts/launch_job.sh # TD3
ALGO=quantile_td3 TASK=kitchen EXPECTILE_Q=False ALPHA=0.02 GUIDE_COEF=0 QF_LAYER_NORM=True bash scripts/launch_job.sh # TD3


# ALGO=quantile_td3 TASK=adroit EXPECTILE_Q=False ALPHA=2.0 GUIDE_COEF=0 QF_LAYER_NORM=True bash scripts/launch_job.sh # TD3
# ALGO=quantile_td3 TASK=adroit EXPECTILE_Q=False ALPHA=0.5 GUIDE_COEF=0 QF_LAYER_NORM=True bash scripts/launch_job.sh # TD3
ALGO=quantile_td3 TASK=adroit EXPECTILE_Q=False ALPHA=0.2 GUIDE_COEF=0 QF_LAYER_NORM=True bash scripts/launch_job.sh # TD3
# ALGO=quantile_td3 TASK=adroit EXPECTILE_Q=False ALPHA=0.02 GUIDE_COEF=0 QF_LAYER_NORM=True bash scripts/launch_job.sh # TD3


ALGO=quantile_td3 TASK=antmaze EXPECTILE_Q=False ALPHA=0 GUIDE_COEF=1 QF_LAYER_NORM=True bash scripts/launch_job.sh # quantile-v crr
ALGO=quantile_td3 TASK=antmaze EXPECTILE_Q=True ALPHA=0 GUIDE_COEF=1 QF_LAYER_NORM=True bash scripts/launch_job.sh
ALGO=quantile_td3  TASK=antmaze EXPECTILE_Q=False ALPHA=0 GUIDE_COEF=1 QF_LAYER_NORM=True FIXED_STD=True ADV_NORM=True bash scripts/launch_job.sh # try to stable quantile-v crr

# stable q on umaze-diverse
ALGO=diff-bc NOTES=pos_rew_BC REW_SCALE=10.0 TASK=antmaze ALPHA=0.0 GUIDE_COEF=0 GUIDE_WARMUP=True QF_LAYER_NORM=True bash scripts/launch_job.sh # TD3 warmup guide
ALGO=diff-bc NOTES=pos_rew_TD3 REW_SCALE=1.0 TASK=antmaze ALPHA=2.0 GUIDE_COEF=0 GUIDE_WARMUP=True QF_LAYER_NORM=True bash scripts/launch_job.sh # TD3 warmup guide
ALGO=diff-bc NOTES=pos_rew_TD3 REW_SCALE=10.0 TASK=antmaze ALPHA=2.0 GUIDE_COEF=0 GUIDE_WARMUP=True QF_LAYER_NORM=True bash scripts/launch_job.sh # TD3 warmup guide
ALGO=diff-bc NOTES=pos_rew_TD3 REW_SCALE=100.0 TASK=antmaze ALPHA=2.0 GUIDE_COEF=0 GUIDE_WARMUP=True QF_LAYER_NORM=True bash scripts/launch_job.sh # TD3 warmup guide
ALGO=diff-bc NOTES=clip_BC REW_BIAS=-1 TARGET_CLIP=True MAX_Q=0 TASK=antmaze ALPHA=0.0 GUIDE_COEF=0 GUIDE_WARMUP=True QF_LAYER_NORM=True bash scripts/launch_job.sh # TD3 warmup guide
ALGO=diff-bc NOTES=trust_reg_BC REW_BIAS=-1 TRUST_REG=True MAX_Q=0 TASK=antmaze ALPHA=0.0 GUIDE_COEF=0 GUIDE_WARMUP=True QF_LAYER_NORM=True bash scripts/launch_job.sh # TD3 warmup guide

ALGO=diff-TD3 NOTES=pos_rew_TD3 REW_SCALE=10.0 TASK=antmaze ALPHA=2.0 GUIDE_COEF=0 GUIDE_WARMUP=True QF_LAYER_NORM=True bash scripts/launch_job.sh
ALGO=diff-TD3 NOTES=pos_rew_TD3 TASK=adroit ALPHA=0.2 GUIDE_COEF=0 QF_LAYER_NORM=True bash scripts/launch_job.sh
ALGO=diff-TD3 NOTES=pos_rew_TD3 TASK=adroit ALPHA=0.05 GUIDE_COEF=0 QF_LAYER_NORM=True bash scripts/launch_job.sh
ALGO=diff-TD3 NOTES=pos_rew_TD3 TASK=adroit ALPHA=0.02 GUIDE_COEF=0 QF_LAYER_NORM=True bash scripts/launch_job.sh

# OPER
ALGO=diff-TD3 TASK=gym ALPHA=2.0 OPER=True  bash scripts/launch_job.sh
ALGO=diff-TD3 TASK=gym ALPHA=2.0 OPER=True TWO_SAMPLER=True  bash scripts/launch_job.sh
ALGO=diff-TD3 TASK=gym ALPHA=2.0 OPER=True PRIORITY=adv TWO_SAMPLER=True  bash scripts/launch_job.sh

ALGO=diff-TD3 TASK=antmaze OPER=True TWO_SAMPLER=True REW_SCALE=10.0  ALPHA=2.0 GUIDE_COEF=0 GUIDE_WARMUP=True QF_LAYER_NORM=True bash scripts/launch_job.sh
ALGO=diff-TD3 TASK=antmaze OPER=True TWO_SAMPLER=True PRIORITY=adv REW_SCALE=10.0  ALPHA=2.0 GUIDE_COEF=0 GUIDE_WARMUP=True QF_LAYER_NORM=True bash scripts/launch_job.sh
ALGO=diff-TD3 NOTES=pbase TASK=antmaze OPER=True TWO_SAMPLER=True REW_SCALE=10.0  ALPHA=2.0 GUIDE_COEF=0 GUIDE_WARMUP=True QF_LAYER_NORM=True bash scripts/launch_job_slurm.sh

# dist rl
ALGO=quantile_td3 TASK=gym ALPHA=2.0 GUIDE_COEF=0 QF_LAYER_NORM=True DIST_RL=True bash scripts/launch_job_slurm.sh
eval "$(GPU=8 ALGO=diff-TD3 NOTES=dist-rl TASK=antmaze REW_SCALE=10.0 ALPHA=2.0 GUIDE_COEF=0 GUIDE_WARMUP=True QF_LAYER_NORM=True DIST_RL=True  bash scripts/launch_job.sh)"   

# 
ALGO=quantile_td3 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.1 QF_LAYER_NORM=True bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0 QF_LAYER_NORM=True bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0 QF_LAYER_NORM=False bash scripts/launch_job_slurm.sh

eval "$(GPU=1 ALGO=quantile_td3 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_ANNEAL=False QF_LAYER_NORM=True bash scripts/launch_job.sh)"

ALGO=quantile_td3 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 LB_RATE=10 bash scripts/launch_job_slurm.sh

ALGO=quantile_td3 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.1 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=1.0 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 bash scripts/launch_job_slurm.sh

eval "$(GPU=0 ALGO=quantile_td3 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 bash scripts/launch_job.sh)"
eval "$(GPU=1 ALGO=quantile_td3 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=0.1 bash scripts/launch_job.sh)"
eval "$(GPU=1 ALGO=quantile_td3 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01  bash scripts/launch_job.sh)"
eval "$(GPU=0 ALGO=quantile_td3 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=1.0  bash scripts/launch_job.sh)"
eval "$(GPU=0 ALGO=quantile_td3 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=1.0 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 bash scripts/launch_job.sh)"


ALGO=quantile_td3 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=1.0 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 RESET_Q=True RESET_INTERVAL=1000000 RESET_ACTOR=False MAX_TGT_Q=True bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=1.0 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 RESET_Q=True RESET_INTERVAL=1000000 RESET_ACTOR=True MAX_TGT_Q=True bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 RESET_Q=True RESET_INTERVAL=250000 RESET_ACTOR=False MAX_TGT_Q=True bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 RESET_Q=True RESET_INTERVAL=250000 RESET_ACTOR=True MAX_TGT_Q=True bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 RESET_Q=True RESET_INTERVAL=500000 RESET_ACTOR=False MAX_TGT_Q=True bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 RESET_Q=True RESET_INTERVAL=500000 RESET_ACTOR=True MAX_TGT_Q=True bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 RESET_Q=True RESET_INTERVAL=1000000 RESET_ACTOR=False MAX_TGT_Q=True bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 RESET_Q=True RESET_INTERVAL=1000000 RESET_ACTOR=True MAX_TGT_Q=True bash scripts/launch_job_slurm.sh

ALGO=quantile_td3 NOTES=stop_q_tgt TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=1.0 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 RESET_Q=True RESET_INTERVAL=1000000 RESET_ACTOR=True MAX_TGT_Q=True bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=stop_q_tgt TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 RESET_Q=True RESET_INTERVAL=1000000 RESET_ACTOR=True MAX_TGT_Q=True bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=stop_q_tgt TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 RESET_Q=True RESET_INTERVAL=500000 RESET_ACTOR=True MAX_TGT_Q=True bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=stop_q_tgt TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 RESET_Q=True RESET_INTERVAL=250000 RESET_ACTOR=True MAX_TGT_Q=True bash scripts/launch_job_slurm.sh

ALGO=quantile_td3 NOTES=stop_q_tgt TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=1.0 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 RESET_Q=True RESET_INTERVAL=1000000 RESET_ACTOR=False MAX_TGT_Q=True bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=stop_q_tgt TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 RESET_Q=True RESET_INTERVAL=1000000 RESET_ACTOR=False MAX_TGT_Q=True bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=stop_q_tgt TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 RESET_Q=True RESET_INTERVAL=500000 RESET_ACTOR=False MAX_TGT_Q=True bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=stop_q_tgt TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 RESET_Q=True RESET_INTERVAL=250000 RESET_ACTOR=False MAX_TGT_Q=True bash scripts/launch_job_slurm.sh



ALGO=quantile_td3 NOTES=L2_norm TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 WEIGHT_DECAY=0.1 bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=L2_norm TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 WEIGHT_DECAY=0.05 bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=L2_norm TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 WEIGHT_DECAY=0.01 bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=L2_norm TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 WEIGHT_DECAY=0.001 bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=L2_norm TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 WEIGHT_DECAY=0.0001 bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=L2_norm TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 WEIGHT_DECAY=0.00001 bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=L2_norm TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 WEIGHT_DECAY=0.000001 bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=L2_norm TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 WEIGHT_DECAY=0 bash scripts/launch_job_slurm.sh

ALGO=quantile_td3 NOTES=L2_norm TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 WEIGHT_DECAY=0 bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=L2_norm TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=1.0 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 WEIGHT_DECAY=0 bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=L2_norm TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=1.0 NORM_REW=True REW_SCALE=10 WEIGHT_DECAY=0 bash scripts/launch_job_slurm.sh


ALGO=quantile_td3 NOTES=eas_sweep TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 bash scripts/launch_job_slurm.sh

ALGO=quantile_td3 NOTES=data_aug_v2 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 STATE_SIGMA=0 ACTION_SIGMA=0.0001 bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=data_aug_v2 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 STATE_SIGMA=0 ACTION_SIGMA=0.0003 bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=data_aug_v2 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 STATE_SIGMA=0 ACTION_SIGMA=0.001 bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=data_aug_v2 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 STATE_SIGMA=0 ACTION_SIGMA=0.003 bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=data_aug_v2 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 STATE_SIGMA=0 ACTION_SIGMA=0.01 bash scripts/launch_job_slurm.sh

ALGO=quantile_td3 NOTES=data_aug_v2 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 STATE_SIGMA=0.0001 ACTION_SIGMA=0 bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=data_aug_v2 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 STATE_SIGMA=0.0003 ACTION_SIGMA=0 bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=data_aug_v2 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 STATE_SIGMA=0.001 ACTION_SIGMA=0 bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=data_aug_v2 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 STATE_SIGMA=0.003 ACTION_SIGMA=0 bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=data_aug_v2 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.01 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 STATE_SIGMA=0.01 ACTION_SIGMA=0 bash scripts/launch_job_slurm.sh


ALGO=quantile_td3 NOTES=actor_L2_norm TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=1.0 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 WEIGHT_DECAY=0.01 bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=actor_L2_norm TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=1.0 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 WEIGHT_DECAY=0.001 bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=actor_L2_norm TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=1.0 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 WEIGHT_DECAY=0.0001 bash scripts/launch_job_slurm.sh
ALGO=quantile_td3 NOTES=actor_L2_norm TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=1.0 QF_LAYER_NORM=True NORM_REW=True REW_SCALE=10 WEIGHT_DECAY=0.00001 bash scripts/launch_job_slurm.sh


ALGO=diff-TD3 NOTES=pos_rew_TD3 REW_SCALE=10.0 TASK=antmaze ALPHA=2.0 DIFF_COEF=1.0 GUIDE_COEF=0 GUIDE_WARMUP=False QF_LAYER_NORM=True bash scripts/launch_job_slurm.sh
ALGO=diff-TD3 NOTES=pos_rew_TD3 REW_SCALE=10.0 TASK=antmaze ALPHA=2.0 DIFF_COEF=0.5 GUIDE_COEF=0 GUIDE_WARMUP=False QF_LAYER_NORM=True bash scripts/launch_job_slurm.sh
ALGO=diff-TD3 NOTES=pos_rew_TD3 REW_SCALE=10.0 TASK=antmaze ALPHA=2.0 DIFF_COEF=0.1 GUIDE_COEF=0 GUIDE_WARMUP=False QF_LAYER_NORM=True bash scripts/launch_job_slurm.sh
ALGO=diff-TD3 NOTES=pos_rew_TD3 REW_SCALE=10.0 TASK=antmaze ALPHA=2.0 DIFF_COEF=0 GUIDE_COEF=0 GUIDE_WARMUP=False QF_LAYER_NORM=True bash scripts/launch_job_slurm.sh

ALGO=diff-TD3 NOTES=pos_rew_TD3 REW_SCALE=10.0 TASK=antmaze ALPHA=2.0 DIFF_COEF=0.1 GUIDE_COEF=0 GUIDE_WARMUP=False QF_LAYER_NORM=True bash scripts/launch_job_slurm.sh


eval "$(GPU=0 ALGO=diff-TD3 NOTES=pos_rew_TD3 REW_SCALE=10.0 TASK=antmaze ALPHA=2.0 DIFF_COEF=0.1 GUIDE_COEF=0 GUIDE_WARMUP=False QF_LAYER_NORM=True ONLY_PENU_NORM=True bash scripts/launch_job.sh)"
eval "$(GPU=1 ALGO=quantile_td3 TASK=gym ALPHA=2.0 GUIDE_COEF=0 DIFF_COEF=0.1 QF_LAYER_NORM=True ONLY_PENU_NORM=True NORM_REW=True REW_SCALE=10 bash scripts/launch_job.sh)"
