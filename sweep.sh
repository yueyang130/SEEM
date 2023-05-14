TAG=layer_norm REW_SCALE=10.0 TASK=antmaze ALPHA=2.5 GUIDE_COEF=1.0 QF_LAYER_NORM=1 bash launch_job_slurm.sh

TAG=layer_norm REW_SCALE=1.0 REW_BIAS=-1 TASK=antmaze ALPHA=2.5 GUIDE_COEF=1.0 QF_LAYER_NORM=1 bash launch_job_slurm.sh
TAG=layer_norm REW_SCALE=10.0 TASK=antmaze ALPHA=2.5 GUIDE_COEF=1.0 QF_LAYER_NORM=0 bash launch_job_slurm.sh



TAG=online_per_v2 TASK=gym ALPHA=2.5 GUIDE_COEF=1.0 RESAMPLE=True ONLINE_PER=1 PER_TEMP=1.0 bash launch_job_slurm.sh
TAG=online_per_v2 TASK=gym ALPHA=2.5 GUIDE_COEF=1.0 RESAMPLE=True ONLINE_PER=1 PER_TEMP=0.8 bash launch_job_slurm.sh
TAG=online_per_v2 TASK=gym ALPHA=2.5 GUIDE_COEF=1.0 RESAMPLE=True ONLINE_PER=1 PER_TEMP=0.6 bash launch_job_slurm.sh
TAG=online_per_v2 TASK=gym ALPHA=2.5 GUIDE_COEF=1.0 RESAMPLE=True ONLINE_PER=1 PER_TEMP=0.4 bash launch_job_slurm.sh
TAG=online_per_v2 TASK=gym ALPHA=2.5 GUIDE_COEF=1.0 RESAMPLE=True ONLINE_PER=1 PER_TEMP=0.2 bash launch_job_slurm.sh
TAG=online_per_v2 TASK=gym ALPHA=2.5 GUIDE_COEF=1.0 RESAMPLE=True ONLINE_PER=1 PER_TEMP=0 bash launch_job_slurm.sh

TAG=online_per_v2 TASK=gym ALPHA=2.5 GUIDE_COEF=1.0 RESAMPLE=True ONLINE_PER=1 PER_TEMP=0.6 START=2 RUNS=3 bash launch_job_slurm.sh


TAG=similarity TASK=gym ALPHA=2.5 BC_COEF=0 bash launch_job_slurm.sh
TAG=similarity TASK=antmaze ALPHA=2.5 BC_COEF=0 REW_SCALE=10.0 bash launch_job_slurm.sh

TAG=similarity_v2 TASK=gym ALPHA=2.5 BC_COEF=0 bash launch_job_slurm.sh

TAG=similarity_v2 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 bash launch_job_slurm.sh
TAG=similarity_v2 TASK=antmaze ALPHA=2.5 BC_COEF=0 TAU=1.0 REW_SCALE=10.0 bash  launch_job_slurm.sh

TAG=similarity_v2_1000 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 bash launch_job_slurm.sh

TAG=similarity_ntk TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 OPTIMIZER=sgd bash launch_job_slurm.sh
TAG=similarity_ntk TASK=gym ALPHA=2.5 BC_COEF=0 TAU=0.005 OPTIMIZER=sgd bash launch_job_slurm.sh
TAG=similarity_ntk TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 OPTIMIZER=adam bash launch_job_slurm.sh


TAG=similarity_grad_v2 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 OPTIMIZER=adam bash launch_job_slurm.sh


eval "$(GPU=0 TAG=eigenvalue TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 DISCOUNT=0.95 OPTIMIZER=adam bash launch_job.sh)"
eval "$(GPU=0 TAG=eigenvalue TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 DISCOUNT=0.94 OPTIMIZER=adam bash launch_job.sh)"
eval "$(GPU=0 TAG=eigenvalue TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 DISCOUNT=0.93 OPTIMIZER=adam bash launch_job.sh)"
eval "$(GPU=1 TAG=eigenvalue TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 DISCOUNT=0.92 OPTIMIZER=adam bash launch_job.sh)"
eval "$(GPU=1 TAG=eigenvalue TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 DISCOUNT=0.91 OPTIMIZER=adam bash launch_job.sh)"
eval "$(GPU=1 TAG=eigenvalue TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 DISCOUNT=0.90 OPTIMIZER=adam bash launch_job.sh)"


eval "$(GPU=0 TAG=eigenvalue_v2 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 DISCOUNT=0.99 OPTIMIZER=adam bash launch_job.sh)"
eval "$(GPU=0 TAG=eigenvalue_v2 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 DISCOUNT=0.95 OPTIMIZER=adam bash launch_job.sh)"
eval "$(GPU=0 TAG=eigenvalue_v2 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 DISCOUNT=0.93 OPTIMIZER=adam bash launch_job.sh)"

eval "$(GPU=0 TAG=eigenvalue_v2 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 DISCOUNT=0.92 OPTIMIZER=adam bash launch_job.sh)"
eval "$(GPU=1 TAG=eigenvalue_v2 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 DISCOUNT=0.91 OPTIMIZER=adam bash launch_job.sh)"
eval "$(GPU=1 TAG=eigenvalue_v2 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 DISCOUNT=0.90 OPTIMIZER=adam bash launch_job.sh)"
eval "$(GPU=1 TAG=eigenvalue_v2 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 DISCOUNT=0.85 OPTIMIZER=adam bash launch_job.sh)"
eval "$(GPU=1 TAG=eigenvalue_v2 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 DISCOUNT=0 OPTIMIZER=adam bash launch_job.sh)"


TAG=similarity_final_v2 START=1 RUNS=3 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 OPTIMIZER=adam bash launch_job_slurm.sh
TAG=similarity_final_v2 START=1 RUNS=3 TASK=antmaze ALPHA=2.5 BC_COEF=0 TAU=1.0 OPTIMIZER=adam bash launch_job_slurm.sh
