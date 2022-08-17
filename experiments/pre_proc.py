import absl.app
import absl.flags
import os

import gym
import numpy as np
import chex
import pickle

import algos
import json
import tqdm

from data import Dataset, RandSampler, SlidingWindowSampler, RLUPDataset, DM2Gym, BalancedSampler
from utilities.jax_utils import batch_to_jax
from utilities.replay_buffer import get_d4rl_dataset, ReplayBuffer
from utilities.sampler import TrajSampler, StepSampler
from utilities.utils import (
  # SOTALogger,
  WandBLogger,
  Timer,
  define_flags_with_default,
  get_user_flags,
  prefix_metrics,
  set_random_seed,
  normalize,
  norm_obs
)
from viskit.logging import logger, setup_logger
from pathlib import Path
from flax import linen as nn
import tqdm

SOTALogger = WandBLogger

FLAGS_DEF = define_flags_with_default(
  env='walker2d-medium-v2',
  algo='IQL',
  max_traj_length=1000,
  seed=42,
  save_model=False,
  batch_size=256,
  reward_scale=1,
  reward_bias=0,
  clip_action=0.999,
  encoder_arch='64-64',
  policy_arch='256-256',
  qf_arch='256-256',
  orthogonal_init=False,
  policy_log_std_multiplier=1.0,
  policy_log_std_offset=-1.0,
  # n_epochs=2000,
  n_epochs=1,
  bc_epochs=0,
  n_train_step_per_epoch=1000,
  eval_period=10,
  eval_n_trajs=1,
  # configs for trining scheme
  sampler='random', # online, random, balanced
  window_size=3000,  # window size for online sampler
  logging=SOTALogger.get_default_config(),
  dataset='d4rl',
  rl_unplugged_task_class = 'control_suite',
  use_resnet=False,
  use_layer_norm=False,
  activation='elu',
  replay_buffer_size=1000000,
  n_env_steps_per_epoch=1000,
  online=False,
  normalize_reward=False,
  embedding_dim=64,
  decoupled_q=False,
  ibal=True,
  cql_n_actions=50,
  render=True,
  run_id='',
  model_id='final'
)


def main(argv):
  FLAGS = absl.flags.FLAGS
  off_algo = getattr(algos, FLAGS.algo)
  algo_cfg = off_algo.get_default_config()

  algo_cfg['ibal'] = FLAGS.ibal
  algo_cfg['cql_n_actions'] = FLAGS.cql_n_actions

  variant = get_user_flags(FLAGS, FLAGS_DEF)
  for k, v in algo_cfg.items():
    variant[f'algo.{k}'] = v
  
  logging_configs = FLAGS.logging

  is_adroit = any([w in FLAGS.env for w in ['pen', 'hammer', 'door', 'relocate']])
  is_kitchen = 'kitchen' in FLAGS.env
  is_mujoco = any([w in FLAGS.env for w in ['hopper', 'walker', 'cheetah']])
  is_antmaze = 'ant' in FLAGS.env

  env_log_name = ''
  env_render_fn = 'render'
  if is_adroit:
    env_log_name = 'Adroit'
    env_render_fn = 'mj_render'
  elif is_kitchen:
    env_log_name = 'Kitchen'
  elif is_mujoco:
    env_log_name = 'Mujoco'
  elif is_antmaze:
    env_log_name = 'AntMaze'
  else:
    raise NotImplementedError

  logging_configs['project'] = f"{FLAGS.algo}-{env_log_name}"
  sota_logger = SOTALogger(
    config=logging_configs, variant=variant, env_name=FLAGS.env
  )
  setup_logger(
    variant=variant,
    exp_id=sota_logger.experiment_id,
    seed=FLAGS.seed,
    base_log_dir=FLAGS.logging.output_dir,
    include_exp_prefix_sub_dir=False
  )

  set_random_seed(FLAGS.seed)
  obs_mean = 0
  obs_std = 1
  obs_clip = np.inf

  tasks = ['pen', 'hammer', 'door', 'relocate']
  alternatives = ['human', 'cloned', 'expert']

  statistics = dict()

  for t in tqdm.tqdm(tasks):
    for a in alternatives:
      env_name = f"{t}-{a}-v0"
      dataset = get_d4rl_dataset(
        gym.make(env_name), 1, 1
      )

      obs_mean = dataset['observations'].mean()
      obs_std = dataset['observations'].std()
      obs_clip = 10

      statistics[env_name] = (obs_mean, obs_std, obs_clip)
  
  for t in tqdm.tqdm(['complete', 'partial', 'mixed']):
    env_name = f"kitchen-{t}-v0"
    dataset = get_d4rl_dataset(
      gym.make(env_name), 1, 1
    )

    obs_mean = dataset['observations'].mean()
    obs_std = dataset['observations'].std()
    obs_clip = 10

    data = [obs_mean, obs_std, obs_clip]
    data = [float(d) for d in data]

    statistics[env_name] = data
  
  import pdb;pdb.set_trace()
  with open("statistics.json", 'w') as fout:
    json.dump(statistics, fout, indent=2)
 

if __name__ == '__main__':
  absl.app.run(main)
