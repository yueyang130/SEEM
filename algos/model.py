# Copyright 2022 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
NN models for MISA.
"""

from functools import partial

import distrax
import jax
import jax.numpy as jnp
from flax import linen as nn

from utilities.jax_utils import extend_and_repeat, next_rng

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5
MEAN_MIN = -9.0
MEAN_MAX = 9.0


def update_target_network(main_params, target_params, tau):
  return jax.tree_map(
    lambda x, y: tau * x + (1.0 - tau) * y, main_params, target_params
  )


def multiple_action_q_function(forward):

  def wrapped(self, observations, actions, **kwargs):
    multiple_actions = False
    batch_size = observations.shape[0]
    if actions.ndim == 3 and observations.ndim == 2:
      multiple_actions = True
      observations = extend_and_repeat(observations, 1, actions.shape[1])
      observations = observations.reshape(-1, observations.shape[-1])
      actions = actions.reshape(-1, actions.shape[-1])
    q_values = forward(self, observations, actions, **kwargs)  # (batch_size * repeat, num_atoms)
    if multiple_actions:
      num_atoms = q_values.shape[-1]
      q_values = q_values.reshape(batch_size, -1, num_atoms)
    return q_values

  return wrapped


class Scalar(nn.Module):
  init_value: float

  def setup(self):
    self.value = self.param("value", lambda x: self.init_value)

  def __call__(self):
    return self.value


class FullyConnectedNetwork(nn.Module):
  output_dim: int
  arch: str = "256-256"
  orthogonal_init: bool = False
  use_layer_norm: bool = False
  activation: str = "relu"

  @nn.compact
  def __call__(self, input_tensor):
    x = input_tensor
    hidden_sizes = [int(h) for h in self.arch.split("-")]
    for h in hidden_sizes:
      if self.orthogonal_init:
        x = nn.Dense(
          h,
          kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
          bias_init=jax.nn.initializers.zeros,
        )(
          x
        )
      else:
        x = nn.Dense(h)(x)
      if self.use_layer_norm:
        x = nn.LayerNorm()(x)
      x = getattr(nn, self.activation)(x)

    if self.orthogonal_init:
      output = nn.Dense(
        self.output_dim,
        kernel_init=jax.nn.initializers.orthogonal(1e-2),
        bias_init=jax.nn.initializers.zeros,
      )(
        x
      )
    else:
      output = nn.Dense(
        self.output_dim,
        kernel_init=jax.nn.initializers.variance_scaling(
          1e-2, "fan_in", "uniform"
        ),
        bias_init=jax.nn.initializers.zeros,
      )(
        x
      )

    if self.use_layer_norm:
      x = nn.LayerNorm()(x)

    return output


class FullyConnectedQFunction(nn.Module):
  observation_dim: int
  action_dim: int
  arch: str = "256-256"
  orthogonal_init: bool = False
  use_layer_norm: bool = False
  activation: str = "relu"

  @nn.compact
  @multiple_action_q_function
  def __call__(self, observations, actions):
    x = jnp.concatenate([observations, actions], axis=-1)
    x = FullyConnectedNetwork(
      output_dim=1,
      arch=self.arch,
      orthogonal_init=self.orthogonal_init,
      use_layer_norm=self.use_layer_norm,
      activation=self.activation,
    )(
      x
    )
    return jnp.squeeze(x, -1)

  @property
  def input_size(self):
    return self.observation_dim


class FullyConnectedVFunction(nn.Module):
  observation_dim: int
  arch: str = '256-256'
  orthogonal_init: bool = False
  use_layer_norm: bool = False
  activation: str = 'relu'

  @nn.compact
  def __call__(self, observations):
    x = FullyConnectedNetwork(
      output_dim=1,
      arch=self.arch,
      orthogonal_init=self.orthogonal_init,
      use_layer_norm=self.use_layer_norm,
      activation=self.activation
    )(
      observations
    )

    return jnp.squeeze(x, -1)

  @property
  def input_size(self):
    return self.observation_dim


class TanhGaussianPolicy(nn.Module):
  observation_dim: int
  action_dim: int
  arch: str = "256-256"
  orthogonal_init: bool = False
  log_std_multiplier: float = 1.0
  log_std_offset: float = -1.0
  use_layer_norm: bool = False

  def setup(self):
    self.base_network = FullyConnectedNetwork(
      output_dim=2 * self.action_dim,
      arch=self.arch,
      orthogonal_init=self.orthogonal_init,
      use_layer_norm=self.use_layer_norm,
    )
    self.log_std_multiplier_module = Scalar(self.log_std_multiplier)
    self.log_std_offset_module = Scalar(self.log_std_offset)

  def log_prob(self, observations, actions):
    if actions.ndim == 3:
      observations = extend_and_repeat(observations, 1, actions.shape[1])
    base_network_output = self.base_network(observations)
    mean, log_std = jnp.split(base_network_output, 2, axis=-1)

    # TODO (max): disable if we want to revert to the original model
    # This is taken from the original CQL implimentation
    mean = jnp.clip(mean, MEAN_MIN, MEAN_MAX)
    log_std = (
      self.log_std_multiplier_module() * log_std + self.log_std_offset_module()
    )

    # TODO (max): disable if we want to revert to the original model
    # This is taken from the original CQL implimentation
    # The original LOG_SIG_MIN = -20, LOG_SIG_MAX = 2.0
    log_std = jnp.clip(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
    action_distribution = distrax.Transformed(
      distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)),
      distrax.Block(distrax.Tanh(), ndims=1),
    )
    return action_distribution.log_prob(actions)

  def __call__(self, rng, observations, deterministic=False, repeat=None):
    if repeat is not None:
      observations = extend_and_repeat(observations, 1, repeat)
    base_network_output = self.base_network(observations)
    mean, log_std = jnp.split(base_network_output, 2, axis=-1)
    mean = jnp.clip(mean, MEAN_MIN, MEAN_MAX)

    log_std = (
      self.log_std_multiplier_module() * log_std + self.log_std_offset_module()
    )
    log_std = jnp.clip(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
    action_distribution = distrax.Transformed(
      distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)),
      distrax.Block(distrax.Tanh(), ndims=1),
    )
    if deterministic:
      samples = jnp.tanh(mean)
      log_prob = action_distribution.log_prob(samples)
    else:
      samples, log_prob = action_distribution.sample_and_log_prob(seed=rng)

    return samples, log_prob

  @property
  def input_size(self):
    return self.observation_dim


class SamplerPolicy(object):

  def __init__(self, policy, policy_params, mean=0, std=1):
    self.policy = policy
    self.policy_params = policy_params
    self.mean = mean
    self.std = std

  def update_params(self, params):
    self.params = params
    return self

  @partial(jax.jit, static_argnames=("self", "deterministic"))
  def act(self, params, rng, observations, deterministic):
    return self.policy.apply(
      params, rng, observations, deterministic, repeat=None
    )

  def __call__(self, observations, deterministic=False):
    observations = (observations - self.mean) / self.std
    actions = self.act(
      self.params, next_rng(), observations, deterministic=deterministic
    )
    if isinstance(actions, tuple):
      actions = actions[0]
    assert jnp.all(jnp.isfinite(actions))
    return jax.device_get(actions)


class DirectMappingPolicy(nn.Module):
  observation_dim: int
  action_dim: int
  max_action: int
  arch: str = '256-256'
  orthogonal_init: bool = False

  def setup(self):
    self.base_network = FullyConnectedNetwork(
      output_dim=self.action_dim,
      arch=self.arch,
      orthogonal_init=self.orthogonal_init
    )

  def __call__(self, rng, observations, deterministic=True, repeat=None):
    # `rng` and `deterministic` are ununsed parameters
    if repeat is not None:
      observations = extend_and_repeat(observations, 1, repeat)
    action = self.base_network(observations)
    return jnp.tanh(action) * self.max_action