
# Efficient Diffusion policy

To run
```bash
PRIORITY=high NS=offbench make run cmd="python -m diffusion.trainer --env=hopper-medium-v2 --algo_cfg.use_pred_astart=True --logging.online=True"
```

`--algo_cfg.use_pred_astart=True` is our efficent training version

`--algo_cfg.use_pred_astart=False` is the orignial version

To reproduce CRR
```bash
PRIORITY=high NS=offbench make run cmd="python -m diffusion.trainer --env=hopper-medium-v2 --algo_cfg.use_pred_astart=True --logging.online=True --algo_cfg.loss_type=CRR --algo_cfg.guide_coef=0.01"
```

Produce IQL
```bash
python -m diffusion.trainer  --logging.output_dir=./experiment_output --logging.online --algo=DiffIQL --obs_norm=False --algo_cfg.loss_type=IQL --sample_method=dpm --algo_cfg.crr_avg_fn=mean --algo_cfg.crr_fn=exp --algo_cfg.adv_norm=False --qf_layer_norm=False --policy_layer_norm=False --algo_cfg.num_timesteps=1000 --algo_cfg.guide_coef=1.0 --norm_reward=True --algo_cfg.lr_decay=True --algo_cfg.fixed_std=False --seed=1 --env=antmaze-umaze-v0 --eval_n_trajs=100 --eval_period=50 --n_epochs=2000 --algo_cfg.max_q_backup=True --algo_cfg.expectile=0.9 --algo_cfg.awr_temperature=10.0
```

RUN IQL
```bash
TASK=antmaze PRIORITY=high LOSS_TYPE=IQL FIXED_STD=False bash scripts/launch_job.sh
```

To run crr+td3
```bash
python -m diffusion.trainer  --logging.output_dir=./ckpts --algo=DiffQL --obs_norm=False --algo_cfg.loss_type=Rainbow --sample_method=dpm --algo_cfg.crr_avg_fn=mean --algo_cfg.crr_fn=exp --algo_cfg.crr_adv_norm=False --qf_layer_norm=False --policy_layer_norm=False --algo_cfg.num_timesteps=1000 --seed=1 --env=walker2d-medium-v2  --algo_cfg.crr_weight_mode=mle --algo_cfg.diff_coef=1.0
```


To run iql+td3
```bash

```

DEBUG Docker `NS=offbench make run cmd="sleep 100000"`

# iql
NOTES=diff-iql TASK=antmaze ALPHA=0 GUIDE_COEF=1.0 START=4 RUNS=5 bash scripts/launch_job.sh
NOTES=diff-iql TASK=antmaze ALPHA=0.25 GUIDE_COEF=1.0 START=3 RUNS=5 bash scripts/launch_job.sh



# expectile td3
NOTES=diff-iql+td3 TASK=gym ALPHA=0.25 GUIDE_COEF=1.0 bash scripts/launch_job.sh
NOTES=diff-iql+td3 TASK=gym ALPHA=0.25 GUIDE_COEF=0 bash scripts/launch_job.sh
NOTES=diff-iql+td3 TASK=gym ALPHA=0 GUIDE_COEF=1.0 bash scripts/launch_job.sh