for level in medium medium-replay medium-expert; do
    for env in halfcheetah hopper walker2d; do
        python -m experiments.vis_dist --env=${env}-${level}-v2 --dataset=d4rl &
    done
done

wait

for env in antmaze-umaze-v2 antmaze-umaze-diverse-v2 antmaze-medium-play-v2 \
    antmaze-medium-diverse-v2 antmaze-large-play-v2 antmaze-large-diverse-v2\
    pen-human-v0 hammer-human-v0 door-human-v0 relocate-human-v0 pen-cloned-v0 hammer-cloned-v0 door-cloned-v0 relocate-cloned-v0\
    kitchen-complete-v0 kitchen-partial-v0 kitchen-undirected-v0; do
    python -m experiments.vis_dist --env=${env} --dataset=d4rl &
done

wait


# for env in antmaze-umaze-v2 antmaze-umaze-diverse-v2 antmaze-medium-play-v2 \
#     antmaze-medium-diverse-v2 antmaze-large-play-v2 antmaze-large-diverse-v2; do
#     # for sampler in  random balanced; do
#     for sampler in  random; do
#         for seed in 1 2 3 4; do
#             PRIORITY=high NS=offrl make run cmd="python -m experiments.main \
#                 --env=${env} --seed=$seed \
#                 --algo=ConservativeSAC  --sampler=$sampler \
#                 --dataset=d4rl \
#                 --cql.cql_min_q_weight=5.0 \
#                 --cql.cql_max_target_backup=True \
#                 --cql.cql_target_action_gap=0.2 \
#                 --orthogonal_init=True \
#                 --cql.cql_lagrange=True \
#                 --cql.cql_temp=1.0 \
#                 --cql.policy_lr=1e-4 \
#                 --cql.qf_lr=3e-4 \
#                 --cql.cql_clip_diff_min=-200 \
#                 --reward_scale=10.0 \
#                 --reward_bias=-5.0 \
#                 --policy_arch='256-256' \
#                 --qf_arch='256-256-256' \
#                 --policy_log_std_multiplier=0.0 \
#                 --eval_period=50 \
#                 --eval_n_trajs=100 \
#                 --n_epochs=1200 \
#                 --bc_epochs=40 \
#                 --logging.output_dir './experiment_output'"
#                 sleep 10;
#         done
#     done
# done



