"""Networks for diffusion policy."""
from typing import Tuple
from functools import partial

import jax
import flax.linen as nn
import jax.numpy as jnp

from algos.model import multiple_action_q_function
from diffusion.diffusion import GaussianDiffusion
from utilities.jax_utils import extend_and_repeat


def mish(x):
  return x * jnp.tanh(nn.softplus(x))


def sinusoidal_embedding(timesteps, dim, max_period=10000):
  """
  Create sinusoidal timestep embeddings.
  :param timesteps: a 1-D Tensor of N indices, one per batch element.
                    These may be fractional.
  :param dim: the dimension of the output.
  :param max_period: controls the minimum frequency of the embeddings.
  :return: an [N x dim] Tensor of positional embeddings.
  """

  half = dim // 2
  freqs = jnp.exp(
    -jnp.log(max_period) * jnp.arange(half, dtype=jnp.float32) / half
  )
  # args = timesteps[:, None] * freqs[None, :]
  args = jnp.expand_dims(timesteps, axis=-1) * freqs[None, :]
  embd = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
  return embd


class TimeEmbedding(nn.Module):
  embed_size: int
  act: callable = mish

  @nn.compact
  def __call__(self, timesteps):
    x = sinusoidal_embedding(timesteps, self.embed_size)
    x = nn.Dense(self.embed_size * 2)(x)
    x = self.act(x)
    x = nn.Dense(self.embed_size)(x)
    return x


class PolicyNet(nn.Module):
  output_dim: int
  arch: Tuple = (256, 256, 256)
  time_embed_size: int = 16
  act: callable = mish
  use_layer_norm: bool = False

  @nn.compact
  def __call__(self, state, action, t):
    time_embed = TimeEmbedding(self.time_embed_size, self.act)(t)
    x = jnp.concatenate([state, action, time_embed], axis=-1)

    for feat in self.arch:
      x = nn.Dense(feat)(x)
      if self.use_layer_norm:
        x = nn.LayerNorm()(x)
      x = self.act(x)

    x = nn.Dense(self.output_dim)(x)
    return x


class DiffusionPolicy(nn.Module):
  diffusion: GaussianDiffusion
  observation_dim: int
  action_dim: int
  arch: Tuple = (256, 256, 256)
  time_embed_size: int = 16
  act: callable = mish
  use_layer_norm: bool = False

  def setup(self):
    self.base_net = PolicyNet(
      output_dim=self.action_dim,
      arch=self.arch,
      time_embed_size=self.time_embed_size,
      act=self.act,
      use_layer_norm=self.use_layer_norm,
    )

  def __call__(self, rng, observations, deterministic=False, repeat=None):
    if repeat is not None:
      observations = extend_and_repeat(observations, 1, repeat)

    shape = observations.shape[:-1] + (self.action_dim,)

    out = self.diffusion.p_sample_loop(
      rng_key=rng,
      model_forward=partial(self.base_net, observations),
      shape=shape,
      clip_denoised=True,
    )

    # noise = jax.random.normal(rng, shape, dtype=observations.dtype)
    # out = self.base_net(observations, noise, jnp.zeros(shape[:-1], dtype=jnp.int32))
    # out = jnp.clip(out, self.diffusion.min_value, self.diffusion.max_value)
    return out

  def forward(self, observations, actions, t):
    return self.base_net(observations, actions, t)

  def loss(self, rng_key, observations, actions, ts):
    terms = self.diffusion.training_losses(
      rng_key,
      model_forward=partial(self.base_net, observations),
      x_start=actions,
      t=ts
    )
    return terms
    noise = jax.random.normal(rng_key, actions.shape, dtype=actions.dtype)
    out = self.base_net(observations, noise, ts * 0)
    return {'loss': jnp.square(out - actions)}

  @property
  def max_action(self):
    return self.diffusion.max_value


class Critic(nn.Module):
  observation_dim: int
  action_dim: int
  arch: Tuple = (256, 256, 256)
  use_layer_norm: bool = False
  act: callable = mish
  use_layer_norm: bool = False

  @nn.compact
  @multiple_action_q_function
  def __call__(self, observations, actions):
    x = jnp.concatenate([observations, actions], axis=-1)

    for feat in self.arch:
      x = nn.Dense(feat)(x)
      if self.use_layer_norm:
        x = nn.LayerNorm()(x)
      x = self.act(x)

    x = nn.Dense(1)(x)
    return jnp.squeeze(x, -1)

  @property
  def input_size(self):
    return self.observation_dim
