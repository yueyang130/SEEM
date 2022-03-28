# Jax Offline Reinforcement Learning

A simple and modular implementation of offline reinforcement learning algorithms in Jax and Flax.

## Setup

To get a GPU pod, simply run
```bash
make submit-k8s
```

Next, when your pod is ready, use `kubectl` to connect to the pod and run your experiments.

Before you start, make sure to run
```bash
pip install -e .
```

## Run Experiments

You can run CQL experiments using the following command:
```bash
python -m experiments.conservative_sac_main --env 'hopper-medium-v2' --logging.output_dir './experiment_output'
```

Run TD3 + BC
```bash
python -m experiments.td3_main --env 'hopper-medium-v2'
```

## Spawn experiments on K8S
```bash
for env in halfcheetah hopper walker2d
do
  PRIORITY=high NS=game-ai make run cmd="python -m experiments.conservative_sac_main \
    --env=${env}-medium-v2 --logging.output_dir=./experiment_output --logging.online
done
```

## Visualize Experiments
You can visualize the experiment metrics with viskit:
```
python -m viskit './experiment_output'
```
and simply navigate to [http://localhost:5000/](http://localhost:5000/)


## Weights and Biases Online Visualization Integration
This codebase can also log to [W&B online visualization platform](https://wandb.ai/site). To log to W&B, you first need to set your W&B API key environment variable.
Alternatively, you could simply run `wandb login`.
Then you can run experiments with W&B logging turned on:
```
python -m experiments.conservative_sac_main --env 'hopper-medium-v2' --logging.output_dir './experiment_output' --logging.online
```

## Credits
The project heavily borrows from this [Jax CQL implementation](https://github.com/young-geng/JaxCQL).
