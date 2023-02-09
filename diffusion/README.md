
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

To run crr+td3
```bash
python -m diffusion.trainer  --logging.output_dir=./ckpts --algo=DiffQL --obs_norm=False --algo_cfg.loss_type=Rainbow --sample_method=dpm --algo_cfg.crr_avg_fn=mean --algo_cfg.crr_fn=exp --algo_cfg.crr_adv_norm=False --qf_layer_norm=False --policy_layer_norm=False --algo_cfg.num_timesteps=1000 --algo=DiffQL --seed=1 --env=walker2d-medium-v2  --algo_cfg.crr_weight_mode=mle --algo_cfg.diff_coef=1.0
```