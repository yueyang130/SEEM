"""Datasets."""
import numpy as np
import os
from rl_unplugged import dm_control_suite
from acme import types
from acme.tf import utils as tf2_utils
import tensorflow as tf
import d4rl
import gym
from gym import spaces
from dm_env import StepType
import tensorflow_datasets as tfds
from pathlib import Path


class Dataset(object):
  """Dataset."""

  def __init__(self, data: dict) -> None:
    self._data = data
    self._keys = list(data.keys())
    self._sampler = None

  def size(self):
    return len(self._data[self._keys[0]])

  def retrieve(self, indices: np.ndarray):
    "Get a batch of data."
    indexed = {}

    for key in self._keys:
      indexed[key] = self._data[key][indices, ...]
    return indexed

  @property
  def sampler(self):
    assert self._sampler is not None
    return self._sampler

  def set_sampler(self, sampler):
    self._sampler = sampler

  def sample(self):
    assert self._sampler is not None

    indices = self._sampler.sample()
    return self.retrieve(indices)


class RLUPDataset(object):
  """RL Uplugged dataset."""
  def __init__(self, task_class, task_name, dataset_path, num_threads=8, batch_size=256, num_shards=100, shuffle_buffer_size=100, action_clipping=1, sarsa=True) -> None:
    self._batch_size = batch_size
    self._num_shards = num_shards
    self._shuffle_buffer_size = shuffle_buffer_size
    self._action_clipping = action_clipping
    self._use_sarsa = sarsa

    if task_class == 'control_suite':
      self.task = dm_control_suite.ControlSuite(task_name=task_name)
    elif task_class == 'humanoid':
      self.task = dm_control_suite.CmuThirdParty(task_name=task_name)
    elif task_class == 'rodent':
      self.task = dm_control_suite.Rodent(task_name=task_name)
    else:
      raise NotImplementedError

    assert os.path.isdir(dataset_path)
    _ds = dm_control_suite.dataset(
      root_path=dataset_path,
      data_path=self.task.data_path,
      shapes=self.task.shapes,
      num_threads=num_threads,
      batch_size=self._batch_size,
      uint8_features=self.task.uint8_features,
      num_shards=self._num_shards,
      shuffle_buffer_size=self._shuffle_buffer_size,
    )

    def discard_extras(sample):
      return sample._replace(data=sample.data[:5])

    _ds = _ds.map(discard_extras).batch(self._batch_size)
    self._ds = iter(_ds)

  def sample(self):
    data = next(self._ds)
    transitions = types.Transition(*data.data)
    observation = transitions.observation
    next_obs = transitions.next_observation
    reward = transitions.reward
    action = transitions.action
    discount = transitions.discount

    observation = tf2_utils.batch_concat(observation)
    next_obs = tf2_utils.batch_concat(next_obs)

    batch = dict(
      observations=observation,
      next_observations=next_obs,
      rewards=reward,
      actions=action,
      discounts=discount,
      dones=tf.zeros_like(reward),
    )

    batch = tfds.as_numpy(batch)
    batch['actions'] = np.clip(
      batch['actions'], -self._action_clipping, self._action_clipping
    )
    return batch

  @property
  def env(self):
    return self.task.environment


class DM2Gym(object):
  def __init__(self, env) -> None:
    self._env = env
  
  @property
  def action_space(self):
    action_spec = self._env.action_spec()
    return spaces.Box(
      low=action_spec.minimum,
      high=action_spec.maximum,
      shape=action_spec.shape,
      dtype=action_spec.dtype
    )
  
  @property
  def observation_space(self):
    specs = [v for _, v in self._env.observation_spec().items()]
    total_lengths = 0
    for s in specs:
      if len(s.shape) == 0:
        total_lengths += 1
      else:
        total_lengths += s.shape[0]
    return spaces.Box(
      low=-np.inf,
      high=np.inf,
      shape=(total_lengths,)
    )
  
  def step(self, action):
    ts = self._env.step(action)
    obs = ts.observation
    reward = ts.reward
    done = ts.step_type == StepType.LAST
    info = {}
    obs = self._wrap_obs(obs)
    if not reward:
      print("+" * 50)
      print("REWARD IS NONE, SET TO ZERO")
      reward = 0
    return obs, reward, done, info
  
  def reset(self):
    ts = self._env.reset()
    return self._wrap_obs(ts.observation)
  
  def _wrap_obs(self, obs):
    obs = tf2_utils.add_batch_dim(obs)
    obs = tf2_utils.batch_concat(obs)
    return tfds.as_numpy(obs)
  
  def get_normalized_score(self, reward):
    return reward


if __name__ == "__main__":
  ds = RLUPDataset(
    'control_suite',
    "walker_walk",
    "/home/aiops/max/jaxoffrl/data",
    sarsa=False,
    batch_size=16
  )

  data = ds.sample()
  print(data)