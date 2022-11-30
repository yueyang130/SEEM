
# Efficient Diffusion policy

To run
```bash
PRIORITY=high NS=offbench make run cmd="python -m diffusion.trainer --env=hopper-medium-v2 --algo_cfg.use_pred_astart=True --logging.online=True"
```

`--algo_cfg.use_pred_astart=True` is our efficent training version
`--algo_cfg.use_pred_astart=False` is the orignial version

