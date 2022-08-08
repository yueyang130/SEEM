#!/usr/bin/python3

import absl.app
import absl.flags

# import d4rl  # type: ignore
import gym
import numpy as np

from algos.crr import CRR

from data import Dataset, RandSampler, SlidingWindowSampler, RLUPDataset, DM2Gym
from utilities.replay_buffer import get_d4rl_dataset
from utilities.traj_dataset import get_traj_dataset, compute_returns
from utilities.sampler import TrajSampler
from utilities.utils import (
  # SOTALogger,
  WandBLogger,
  define_flags_with_default,
  get_user_flags,
)
from pathlib import Path

import matplotlib.pyplot as plt

# from sklearn.decomposition import PCA
# from openTSNE import TSNE


FLAGS_DEF = define_flags_with_default(
  env='antmaze-large-diverse-v0',
  max_traj_length=1000,
  seed=42,
  save_model=False,
  batch_size=256,
  reward_scale=1.0,
  reward_bias=0.0,
  clip_action=0.999,
  policy_arch='256-256',
  qf_arch='256-256',
  orthogonal_init=False,
  policy_log_std_multiplier=1.0,
  policy_log_std_offset=-1.0,
  n_epochs=2000,
  bc_epochs=0,
  n_train_step_per_epoch=1000,
  eval_period=10,
  eval_n_trajs=5,
  # configs for trining scheme
  online=False,  # use online training
  window_size=3000,  # window size for online sampler
  cql=CRR.get_default_config(),
  dataset='d4rl',
  rl_unplugged_task_class = 'control_suite',
  nstep=1,
  discount=0.99,
)


def main(argv):
  FLAGS = absl.flags.FLAGS

  variant = get_user_flags(FLAGS, FLAGS_DEF)

  if FLAGS.dataset == 'd4rl':
    eval_sampler = TrajSampler(
      gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length
    )

    # Build dataset and sampler
    dataset = get_d4rl_dataset(
      eval_sampler.env,
      FLAGS.nstep,
      FLAGS.discount,
    )
  else:
    raise NotImplementedError
  
  # n-step frame-stack dataset
  obs = dataset['observations']
  act = dataset['actions']
  nstep_rewards = dataset['rewards']

  # traj dataset
  traj_dataset, raw_dataset = get_traj_dataset(eval_sampler.env, sorting=False)
  traj_returns = [compute_returns(traj) for traj in traj_dataset]
  traj_lens = [len(traj) for traj in traj_dataset]
  rewards = np.concatenate([[ts[2] for ts in traj] for traj in traj_dataset])

  # rewards = (rewards - rewards.min()) / (rewards.max() - rewards.min())
  # labels = np.floor((rewards * 10)).astype(np.int32)

  plt.subplot(2,2,1)
  plt.hist(rewards, bins=100)
  plt.title('reward dist')

  plt.subplot(2,2,2)
  plt.hist(nstep_rewards, bins=100)
  plt.title('nstep_reward dist')

  plt.subplot(2,2,3)
  plt.hist(traj_returns, bins=100)
  plt.title('traj_return dist')
  
  plt.subplot(2,2,4)
  plt.hist(traj_lens)
  plt.title('traj_len dist')

  plt.savefig(f'vis_results/{FLAGS.env}_hist.jpg')



  

if __name__ == '__main__':
  absl.app.run(main)
