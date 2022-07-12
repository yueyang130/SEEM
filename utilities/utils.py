import os
import pprint
import random
import tempfile
import time
import uuid
from copy import copy
from socket import gethostname

import absl.flags
import cloudpickle as pickle
import numpy as np
import sota
import wandb
from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
from ml_collections.config_flags import config_flags
from tqdm import tqdm

from utilities.jax_utils import init_rng


def split_into_trajectories(observations, actions, rewards, dones_float, next_observations):
  trajs = [[]]

  for i in tqdm(range(len(observations))):
    trajs[-1].append((observations[i], actions[i], rewards[i],
                      dones_float[i], next_observations[i]))
    if dones_float[i] == 1.0 and i + 1 < len(observations):
      trajs.append([])

  return trajs

def normalize(dataset):
  trajs = split_into_trajectories(
    dataset['observations'],
    dataset['actions'],
    dataset['rewards'],
    dataset['dones'],
    dataset['next_observations'],
  )

  def compute_returns(traj):
    episode_return = 0
    for _, _, rew, _, _ in traj:
      episode_return += rew
    return episode_return

  trajs.sort(key=compute_returns)

  dataset['rewards'] /= compute_returns(trajs[-1]) - compute_returns(trajs[0])
  dataset['rewards'] *= 1000.0


class Timer(object):

  def __init__(self):
    self._time = None

  def __enter__(self):
    self._start_time = time.time()
    return self

  def __exit__(self, exc_type, exc_value, exc_tb):
    self._time = time.time() - self._start_time

  def __call__(self):
    return self._time


class SOTALogger(object):

  @staticmethod
  def get_default_config(updates=None):
    config = ConfigDict()
    config.online = True
    config.prefix = ''
    config.project = 'OfflineRL'
    config.output_dir = '/tmp/JaxCQL'
    config.random_delay = 0
    config.experiment_id = config_dict.placeholder(str)
    config.anonymous = config_dict.placeholder(str)
    config.notes = config_dict.placeholder(str)

    if updates is not None:
      config.update(ConfigDict(updates).copy_and_resolve_references())
    return config

  def __init__(self, config, variant, env_name):
    self._env_name = env_name
    self._step = 0
    self.config = self.get_default_config(config)

    if self.config.experiment_id is None:
      self.config.experiment_id = sota.generate_experiment_name(
        self.config.project
      )

    if self.config.prefix != '':
      self.config.project = '{}--{}'.format(
        self.config.prefix, self.config.project
      )

    if self.config.output_dir == '':
      self.config.output_dir = tempfile.mkdtemp()
    else:
      self.config.output_dir = os.path.join(
        self.config.output_dir, self.config.experiment_id
      )
      os.makedirs(self.config.output_dir, exist_ok=True)

    self._variant = copy(variant)

    if 'hostname' not in self._variant:
      self._variant['hostname'] = gethostname()

    if self.config.random_delay > 0:
      time.sleep(np.random.uniform(0, self.config.random_delay))

    self.run = sota.create_job(
      experiment=f"{self._env_name}_{self.config.experiment_id}",
      project=self.config.project,
      config=self.filter_config_params(self._variant),
      job_type='Train/Eval'
    )

  @staticmethod
  def filter_config_params(config):
    ret = {}
    for k, v in config.items():
      if isinstance(v, float) or isinstance(v, int):
        if not np.isinf(v):
          ret[k] = v
    return ret

  def log(self, *args, **kwargs):
    ret = {k: float(v) for k, v in args[0].items()}
    self.run.log(ret, step=int(self._step))
    self._step += 1

  def save_pickle(self, obj, filename):
    with open(os.path.join(self.config.output_dir, filename), 'wb') as fout:
      pickle.dump(obj, fout)

  @property
  def experiment_id(self):
    return self.config.experiment_id

  @property
  def variant(self):
    return self.config.variant

  @property
  def output_dir(self):
    return self.config.output_dir


class WandBLogger(object):

  @staticmethod
  def get_default_config(updates=None):
    config = ConfigDict()
    config.online = True
    config.prefix = 'JaxCQL'
    config.project = 'sac'
    config.output_dir = './experiment_output'
    config.random_delay = 0.0
    config.experiment_id = config_dict.placeholder(str)
    config.anonymous = config_dict.placeholder(str)
    config.notes = config_dict.placeholder(str)

    if updates is not None:
      config.update(ConfigDict(updates).copy_and_resolve_references())
    return config

  def __init__(self, config, variant, env_name):
    self.config = self.get_default_config(config)

    if self.config.experiment_id is None:
      self.config.experiment_id = uuid.uuid4().hex

    if self.config.prefix != '':
      self.config.project = '{}--{}'.format(
        self.config.prefix, self.config.project
      )

    if self.config.output_dir == '':
      self.config.output_dir = tempfile.mkdtemp()
    else:
      self.config.output_dir = os.path.join(
        self.config.output_dir, self.config.experiment_id
      )
      os.makedirs(self.config.output_dir, exist_ok=True)

    self._variant = copy(variant)

    if 'hostname' not in self._variant:
      self._variant['hostname'] = gethostname()

    if self.config.random_delay > 0:
      time.sleep(np.random.uniform(0, self.config.random_delay))

    self.run = wandb.init(
      reinit=True,
      config=self._variant,
      project=self.config.project,
      dir=self.config.output_dir,
      id=self.config.experiment_id,
      anonymous=self.config.anonymous,
      notes=self.config.notes,
      settings=wandb.Settings(
        start_method="thread",
        _disable_stats=True,
      ),
      mode='online' if self.config.online else 'offline',
    )

  def log(self, *args, **kwargs):
    self.run.log(*args, **kwargs)

  def save_pickle(self, obj, filename):
    with open(os.path.join(self.config.output_dir, filename), 'wb') as fout:
      pickle.dump(obj, fout)

  @property
  def experiment_id(self):
    return self.config.experiment_id

  @property
  def variant(self):
    return self.config.variant

  @property
  def output_dir(self):
    return self.config.output_dir


def define_flags_with_default(**kwargs):
  for key, val in kwargs.items():
    if isinstance(val, ConfigDict):
      config_flags.DEFINE_config_dict(key, val)
    elif isinstance(val, bool):
      # Note that True and False are instances of int.
      absl.flags.DEFINE_bool(key, val, 'automatically defined flag')
    elif isinstance(val, int):
      absl.flags.DEFINE_integer(key, val, 'automatically defined flag')
    elif isinstance(val, float):
      absl.flags.DEFINE_float(key, val, 'automatically defined flag')
    elif isinstance(val, str):
      absl.flags.DEFINE_string(key, val, 'automatically defined flag')
    else:
      raise ValueError('Incorrect value type')
  return kwargs


def set_random_seed(seed):
  np.random.seed(seed)
  random.seed(seed)
  init_rng(seed)


def print_flags(flags, flags_def):
  logging.info(
    'Running training with hyperparameters: \n{}'.format(
      pprint.pformat(
        [
          '{}: {}'.format(key, val)
          for key, val in get_user_flags(flags, flags_def).items()
        ]
      )
    )
  )


def get_user_flags(flags, flags_def):
  output = {}
  for key in flags_def:
    val = getattr(flags, key)
    if isinstance(val, ConfigDict):
      output.update(flatten_config_dict(val, prefix=key))
    else:
      output[key] = val

  return output


def flatten_config_dict(config, prefix=None):
  output = {}
  for key, val in config.items():
    if isinstance(val, ConfigDict):
      output.update(flatten_config_dict(val, prefix=key))
    else:
      if prefix is not None:
        output['{}.{}'.format(prefix, key)] = val
      else:
        output[key] = val
  return output


def prefix_metrics(metrics, prefix):
  return {'{}/{}'.format(prefix, key): value for key, value in metrics.items()}
