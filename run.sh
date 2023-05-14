eval "$(GPU=0 TAG=eigenvalue_v2 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 DISCOUNT=0.929 OPTIMIZER=adam bash launch_job.sh)"
eval "$(GPU=0 TAG=eigenvalue_v2 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 DISCOUNT=0.928 OPTIMIZER=adam bash launch_job.sh)"
wait
eval "$(GPU=0 TAG=eigenvalue_v2 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 DISCOUNT=0.927 OPTIMIZER=adam bash launch_job.sh)"
eval "$(GPU=1 TAG=eigenvalue_v2 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 DISCOUNT=0.926 OPTIMIZER=adam bash launch_job.sh)"
wait
eval "$(GPU=0 TAG=eigenvalue_v2 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 DISCOUNT=0.924 OPTIMIZER=adam bash launch_job.sh)"
eval "$(GPU=1 TAG=eigenvalue_v2 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 DISCOUNT=0.923 OPTIMIZER=adam bash launch_job.sh)"
wait
eval "$(GPU=0 TAG=eigenvalue_v2 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 DISCOUNT=0.922 OPTIMIZER=adam bash launch_job.sh)"
eval "$(GPU=1 TAG=eigenvalue_v2 TASK=gym ALPHA=2.5 BC_COEF=0 TAU=1.0 DISCOUNT=0.921 OPTIMIZER=adam bash launch_job.sh)"
wait