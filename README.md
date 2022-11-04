# MISA

Before you start, make sure to run
```bash
pip install -e .
```

Apart from this, you'll have to setup your MuJoCo environment and key as well. Please follow [D4RL](https://github.com/Farama-Foundation/D4RL) repo and setup the environment accordingly.

## Run Experiments

You can run MISA experiments using the following command:
```bash
python -m experiments.main --env 'walker2d-medium-v2' --logging.output_dir './experiment_output'
```

To reproduce CRR, please run:
```bash
python -m experiments.main --logging.output_dir=./experiment_output --logging.online --algo=CRR --algo_cfg.avg_fn=mean --algo_cfg.crr_fn=exp --algo_cfg.crr_beta=1.0 --use_layer_norm=True --algo_cfg.q_weight_method=min --algo_cfg.use_expectile=True --algo_cfg.exp_tau=0.7 --n_epochs=2000
```

## Weights and Biases Online Visualization Integration
This codebase can also log to [W&B online visualization platform](https://wandb.ai/site). To log to W&B, you first need to set your W&B API key environment variable.
Alternatively, you could simply run `wandb login`.

## Credits
The project heavily borrows from this [Jax CQL implementation](https://github.com/young-geng/JaxCQL).
