from functools import partial
from queue import Full
from typing import Any

import distrax
import jax
import jax.numpy as jnp
from flax import linen as nn

from utilities.jax_utils import extend_and_repeat, next_rng
import tensorflow_probability.substrates.jax as tfp
import tensorflow_probability.substrates.jax.distributions as tfd


def update_target_network(main_params, target_params, tau):
  return jax.tree_multimap(
    lambda x, y: tau * x + (1.0 - tau) * y, main_params, target_params
  )


def multiple_action_q_function(forward):
  # Forward the q function with multiple actions on each state, to be used as a decorator
  def wrapped(self, observations, actions, **kwargs):
    multiple_actions = False
    batch_size = observations.shape[0]
    if actions.ndim == 3 and observations.ndim == 2:
      multiple_actions = True
      observations = extend_and_repeat(observations, 1, actions.shape[1]
                                      ).reshape(-1, observations.shape[-1])
      actions = actions.reshape(-1, actions.shape[-1])
    q_values = forward(self, observations, actions, **kwargs)
    if multiple_actions:
      q_values = q_values.reshape(batch_size, -1)
    return q_values

  return wrapped


def multiple_action_decoupled_q_function(forward):
  # Forward the q function with multiple actions on each state, to be used as a decorator
  def wrapped(self, observations, actions, **kwargs):
    multiple_actions = False
    batch_size = observations.shape[0]
    if actions.ndim == 3 and observations.ndim == 2:
      multiple_actions = True
      observations = extend_and_repeat(observations, 1, actions.shape[1]
                                      ).reshape(-1, observations.shape[-1])
      actions = actions.reshape(-1, actions.shape[-1])
    q_values, rewards, values = forward(self, observations, actions, **kwargs)
    if multiple_actions:
      q_values = q_values.reshape(batch_size, -1)
      rewards = rewards.reshape(batch_size, -1)
      values = values.reshape(batch_size, -1)
    return q_values, rewards, values

  return wrapped


class Scalar(nn.Module):
  init_value: float

  def setup(self):
    self.value = self.param('value', lambda x: self.init_value)

  def __call__(self):
    return self.value


class FullyConnectedNetwork(nn.Module):
  output_dim: int
  arch: str = '256-256'
  orthogonal_init: bool = False
  use_layer_norm: bool = False
  activation: str = 'relu'

  @nn.compact
  def __call__(self, input_tensor):
    x = input_tensor
    hidden_sizes = [int(h) for h in self.arch.split('-')]
    for h in hidden_sizes:
      if self.orthogonal_init:
        x = nn.Dense(
          h,
          kernel_init=jax.nn.initializers.orthogonal(jnp.sqrt(2.0)),
          bias_init=jax.nn.initializers.zeros
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
        bias_init=jax.nn.initializers.zeros
      )(
        x
      )
    else:
      output = nn.Dense(
        self.output_dim,
        kernel_init=jax.nn.initializers.variance_scaling(
          1e-2, 'fan_in', 'uniform'
        ),
        bias_init=jax.nn.initializers.zeros
      )(
        x
      )
    
    if self.use_layer_norm:
      x = nn.LayerNorm()(x)

    return output


class ResidualBlock(nn.Module):
  hidden_dim: int
  norm: Any
  act: Any

  @nn.compact
  def __call__(self, x):
    residual = x
    y = nn.Dense(self.hidden_dim)(x)
    y = self.norm()(y)
    y = self.act(y)
    y = nn.Dense(self.hidden_dim)(y)
    y = self.norm()(y)

    if residual.shape != y.shape:
      residual = nn.Dense(self.hidden_dim)(residual)
      residual = nn.LayerNorm()(residual)

    return self.act(y + residual)


class ResEncoder(nn.Module):
  num_blocks: int
  hidden_dim: int
  act: Any

  @nn.compact
  def __call__(self, x):
    def norm_fn():
      return jax.nn.standardize

    for _ in range(self.num_blocks):
      x = ResidualBlock(hidden_dim=self.hidden_dim, norm=norm_fn, act=self.act)(x)
    
    x = jax.nn.standardize(x)
    x = nn.elu(x)
    x = nn.Dense(self.hidden_dim)(x)
    x = nn.LayerNorm()(x)
    x = nn.tanh(x)
    
    return x


class IdentityEncoder(nn.Module):
  """An Identity mapping encoder."""
  num_blocks: int
  hidden_dim: int
  act: Any

  @nn.compact
  def __call__(self, x):
    return x


class FullyConnectedQFunction(nn.Module):
  observation_dim: int
  action_dim: int
  arch: str = '256-256'
  orthogonal_init: bool = False
  use_layer_norm: bool = False
  activation: str = 'relu'

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
    )(observations)

    return jnp.squeeze(x, -1)
  
  @property
  def input_size(self):
    return self.observation_dim


class DecoupledQFunction(nn.Module):
  observation_dim: int
  action_dim: int
  arch: str = '256-256'
  orthogonal_init: bool = False
  use_layer_norm: bool = False
  activation: str = 'relu'

  @nn.compact
  @multiple_action_decoupled_q_function
  def __call__(self, observations, actions) -> Any:
    x = jnp.concatenate([observations, actions], axis=-1)
    reward = FullyConnectedNetwork(
      output_dim=1,
      arch=self.arch,
      orthogonal_init=self.orthogonal_init,
      use_layer_norm=self.use_layer_norm,
      activation=self.activation
    )(x)
    value = FullyConnectedNetwork(
      output_dim=1,
      arch=self.arch,
      orthogonal_init=self.orthogonal_init,
      use_layer_norm=self.use_layer_norm,
      activation=self.activation
    )(observations)

    reward = nn.tanh(reward)

    return jnp.squeeze(reward + value, -1), reward, value


class ResQFunction(nn.Module):
  observation_dim: int
  action_dim: int
  arch: str = '1024'
  orthogonal_init: bool = False
  num_blocks: int = 4

  @nn.compact
  @multiple_action_q_function
  def __call__(self, x_emb, actions):
    hidden_dim = int(self.arch)
    act_emb = nn.Dense(hidden_dim)(actions)
    act_emb = nn.LayerNorm()(act_emb)
    act_emb = nn.relu(act_emb)

    x = jnp.concatenate((x_emb, act_emb), axis=-1)
    x = nn.Dense(hidden_dim)(x)
    for _ in range(4):
      x = ResidualBlock(
        hidden_dim,
        nn.LayerNorm,
        nn.relu
      )(x)
    
    x = FullyConnectedNetwork(
      output_dim=1, arch=self.arch, orthogonal_init=self.orthogonal_init
    )(x)

    return jnp.squeeze(x, -1)
  
  @property
  def input_size(self):
    hs = [int(h) for h in self.arch.split('-')]
    return hs[0]


class TanhGaussianPolicy(nn.Module):
  observation_dim: int
  action_dim: int
  arch: str = '256-256'
  orthogonal_init: bool = False
  log_std_multiplier: float = 1.0
  log_std_offset: float = -1.0

  def setup(self):
    self.base_network = FullyConnectedNetwork(
      output_dim=2 * self.action_dim,
      arch=self.arch,
      orthogonal_init=self.orthogonal_init
    )
    self.log_std_multiplier_module = Scalar(self.log_std_multiplier)
    self.log_std_offset_module = Scalar(self.log_std_offset)

  def log_prob(self, observations, actions):
    if actions.ndim == 3:
      observations = extend_and_repeat(observations, 1, actions.shape[1])
    base_network_output = self.base_network(observations)
    mean, log_std = jnp.split(base_network_output, 2, axis=-1)
    log_std = self.log_std_multiplier_module(
    ) * log_std + self.log_std_offset_module()
    log_std = jnp.clip(log_std, -20.0, 2.0)
    action_distribution = distrax.Transformed(
      distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)),
      distrax.Block(distrax.Tanh(), ndims=1)
    )
    return action_distribution.log_prob(actions)

  def __call__(self, rng, observations, deterministic=False, repeat=None):
    if repeat is not None:
      observations = extend_and_repeat(observations, 1, repeat)
    base_network_output = self.base_network(observations)
    mean, log_std = jnp.split(base_network_output, 2, axis=-1)
    log_std = self.log_std_multiplier_module(
    ) * log_std + self.log_std_offset_module()
    log_std = jnp.clip(log_std, -20.0, 2.0)
    action_distribution = distrax.Transformed(
      distrax.MultivariateNormalDiag(mean, jnp.exp(log_std)),
      distrax.Block(distrax.Tanh(), ndims=1)
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


class ClipGaussianPolicy(TanhGaussianPolicy):
  observation_dim: int
  action_dim: int
  arch: str = '512-512-256'
  orthogonal_init: bool = False
  log_std_multiplier: float = 1.0
  log_std_offset: float = -1.0
  use_layer_norm: bool = True
  activation: str = 'elu'

  def __call__(self, rng, observations, deterministic=False, repeat=None):
    if repeat is not None:
      observations = extend_and_repeat(observations, 1, repeat)
    base_network_output = self.base_network(observations)
    mean, log_std = jnp.split(base_network_output, 2, axis=-1)
    log_std = self.log_std_multiplier_module(
    ) * log_std + self.log_std_offset_module()
    log_std = jnp.clip(log_std, -20.0, 2.0)
    action_distribution = distrax.MultivariateNormalDiag(
      mean, jnp.exp(log_std)
    )
    if deterministic:
      samples = mean
      log_prob = action_distribution.log_prob(samples)
    else:
      samples, log_prob = action_distribution.sample_and_log_prob(seed=rng)

    samples = jnp.clip(samples, -1, 1)
    return samples, log_prob
  
  def setup(self):
    self.base_network = FullyConnectedNetwork(
      output_dim=2 * self.action_dim,
      arch=self.arch,
      orthogonal_init=self.orthogonal_init,
      use_layer_norm=self.use_layer_norm,
      activation=self.activation,
    )
    self.log_std_multiplier_module = Scalar(self.log_std_multiplier)
    self.log_std_offset_module = Scalar(self.log_std_offset)
 
  def get_tfd_dist(self, observations):
    base_network_output = self.base_network(observations)
    mean, log_std = jnp.split(base_network_output, 2, axis=-1)
    log_std = self.log_std_multiplier_module(
    ) * log_std + self.log_std_offset_module()
    log_std = jnp.clip(log_std, -20.0, 2.0)
    dist = tfd.MultivariateNormalDiag(
      loc=mean,
      scale_diag=jnp.exp(log_std)
    )
    return dist

  def log_prob(self, observations, actions):
    if actions.ndim == 3:
      observations = extend_and_repeat(observations, 1, actions.shape[1])
    base_network_output = self.base_network(observations)
    mean, log_std = jnp.split(base_network_output, 2, axis=-1)
    log_std = self.log_std_multiplier_module(
    ) * log_std + self.log_std_offset_module()
    log_std = jnp.clip(log_std, -20.0, 2.0)
    action_distribution = distrax.MultivariateNormalDiag(
      mean, jnp.exp(log_std)
    )
    return action_distribution.log_prob(actions)
 

class ResTanhGaussianPolicy(TanhGaussianPolicy):
  observation_dim: int
  action_dim: int
  arch: str = '1024'
  orthogonal_init: bool = False
  log_std_multiplier: float = 1.0
  log_std_offset: float = -1.0
  num_blocks: int = 4

  def setup(self):
    self.hidden_dim = int(self.arch)
    layers = []
    for _ in range(self.num_blocks):
      layers.append(
        ResidualBlock(
          self.hidden_dim,
          nn.LayerNorm,
          nn.relu
        )
      )
    layers.append(
      FullyConnectedNetwork(
        2 * self.action_dim,
        self.arch,
        self.orthogonal_init
      )
    )
    self.base_network = nn.Sequential(layers)
    self.log_std_multiplier_module = Scalar(self.log_std_multiplier)
    self.log_std_offset_module = Scalar(self.log_std_offset)
  
  @property
  def input_size(self):
    hs = [int(h) for h in self.arch.split('-')]
    return hs[0]


class ResClipGaussianPolicy(ClipGaussianPolicy):
  observation_dim: int
  action_dim: int
  arch: str = '1024'
  orthogonal_init: bool = False
  log_std_multiplier: float = 1.0
  log_std_offset: float = -1.0
  num_blocks: int = 4

  def setup(self):
    self.hidden_dim = int(self.arch)
    layers = []
    for _ in range(self.num_blocks):
      layers.append(
        ResidualBlock(
          self.hidden_dim,
          nn.LayerNorm,
          nn.relu
        )
      )
    layers.append(
      FullyConnectedNetwork(
        2 * self.action_dim,
        self.arch,
        self.orthogonal_init
      )
    )
    self.base_network = nn.Sequential(layers)
    self.log_std_multiplier_module = Scalar(self.log_std_multiplier)
    self.log_std_offset_module = Scalar(self.log_std_offset)
  
  @property
  def input_size(self):
    hs = [int(h) for h in self.arch.split('-')]
    return hs[0]


class SamplerPolicy(object):

  def __init__(self, policy, policy_params, mean=0, std=1):
    self.policy = policy
    self.policy_params = policy_params
    self.mean = mean
    self.std = std

  def update_params(self, params):
    self.params = params
    return self

  @partial(jax.jit, static_argnames=('self', 'deterministic'))
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


class SamplerPolicyEncoder(object):

  def __init__(self, agent, params, mean=0, std=1):
    self.agent = agent
    self.params = params 
    self.mean = mean
    self.std = std

  def update_params(self, params):
    self.params = params
    return self
  
  @partial(jax.jit, static_argnames=('self', 'deterministic'))
  def act(self, policy_params, encoder_params, rng, observations, deterministic):
    emb = self.agent.encoder.apply(
      encoder_params, observations
    )
    return self.agent.policy.apply(
      policy_params, rng, emb, deterministic, repeat=None
    )

  def __call__(self, observations, deterministic=False):
    observations = (observations - self.mean) / self.std
    actions = self.act(
      self.params['policy'], self.params['encoder'], next_rng(), observations, deterministic=deterministic
    )
    if isinstance(actions, tuple):
      actions = actions[0]
    assert jnp.all(jnp.isfinite(actions))
    return jax.device_get(actions)


class DirectMappingPolicy(nn.Module):
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


class TD3Critic(nn.Module):
  action_dim: int
  arch: str = '256-256'
  orthogonal_init: bool = False

  def setup(self):
    self.q1 = FullyConnectedNetwork(output_dim=1, arch=self.arch)
    self.q2 = FullyConnectedNetwork(output_dim=1, arch=self.arch)

  @multiple_action_q_function
  def __call__(self, observations, actions):
    x = jnp.concatenate([observations, actions], axis=-1)
    q1 = jnp.squeeze(self.q1(x), -1)
    q2 = jnp.squeeze(self.q2(x), -1)

    return q1, q2

  @multiple_action_q_function
  def q1(self, observations, actions):
    x = jnp.concatenate([observations, actions], axis=-1)
    q1 = jnp.squeeze(self.q1(x), -1)

    return q1


if __name__ == "__main__":
  rng = jax.random.PRNGKey(0)

  encoder = ResEncoder(
    4, 1024, nn.elu
  )
  encoder_params = encoder.init(
    rng, jnp.zeros((10, 64))
  )
  
  policy = ResTanhGaussianPolicy(
    64, 6,
  )
  policy_params = policy.init(
    rng, rng, jnp.zeros((10, 64))
  )

  qf = ResQFunction(
    64, 6
  )

  import ipdb; ipdb.set_trace()