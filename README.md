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

## Weights and Biases Online Visualization Integration
This codebase can also log to [W&B online visualization platform](https://wandb.ai/site). To log to W&B, you first need to set your W&B API key environment variable.
Alternatively, you could simply run `wandb login`.

## Credits
The project heavily borrows from this [Jax CQL implementation](https://github.com/young-geng/JaxCQL).
