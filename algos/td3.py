from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from ml_collections import ConfigDict

from algos.model import update_target_network
from core.core_api import Algo
from utilities.jax_utils import mse_loss, next_rng, value_and_multi_grad


class TD3(Algo):

  @staticmethod
  def get_default_config(updates=None):
    config = ConfigDict()
    config.nstep = 1
    config.discount = 0.99
    config.tau = 0.005
    config.policy_noise = 0.2
    config.noise_clip = 0.5
    config.policy_freq = 2
    config.alpha = 2.5
    config.lr = 3e-4

    if updates is not None:
      config.update(ConfigDict(updates).copy_and_resolve_references())
    return config

  def __init__(self, config, actor, critic):
    self.config = self.get_default_config(config)
    self.actor = actor
    self.critic = critic
    self.observation_dim = actor.observation_dim
    self.action_dim = actor.action_dim
    self.policy_noise = self.config.policy_noise * actor.max_action
    self.max_action = actor.max_action

    self._total_steps = 0
    self._train_states = {}

    actor_params = self.actor.init(
      next_rng(),
      next_rng(),
      jnp.zeros((10, self.observation_dim)),
    )
    self._train_states['actor'] = TrainState.create(
      params=actor_params,
      tx=optax.adam(self.config.lr),
      apply_fn=None,
    )

    critic_params = self.critic.init(
      next_rng(),
      jnp.zeros((10, self.observation_dim)),
      jnp.zeros((10, self.action_dim)),
    )
    self._train_states['critic'] = TrainState.create(
      params=critic_params,
      tx=optax.adam(self.config.lr),
      apply_fn=None,
    )

    self._tgt_params = deepcopy(
      {
        'actor': actor_params,
        'critic': critic_params,
      }
    )

    self._model_keys = ('actor', 'critic')

  @partial(jax.jit, static_argnames=('self', 'policy_update'))
  def _train_step(
    self, train_states, tgt_params, rng, batch, policy_update=False
  ):

    def loss_fn(params, tgt_params, rng):
      observations = batch['observations']
      actions = batch['actions']
      rewards = batch['rewards']
      next_observations = batch['next_observations']
      dones = batch['dones']

      rng, split_rng = jax.random.split(rng)
      # Select next actions according to policy and add noise
      noise = jax.random.truncated_normal(
        split_rng,
        -self.config.noise_clip,
        self.config.noise_clip,
        shape=actions.shape,
        dtype=actions.dtype,
      ) * self.policy_noise
      next_action = self.actor.apply(params['actor'], rng, next_observations)
      next_action = jnp.clip(
        next_action + noise, -self.max_action, self.max_action
      )

      # Compute the target Q values (without gradient)
      tgt_q1, tgt_q2 = self.critic.apply(
        tgt_params['critic'], next_observations, next_action
      )
      tgt_q = jnp.minimum(tgt_q1, tgt_q2)
      tgt_q = rewards + (1 - dones) * self.config.discount * tgt_q
      tgt_q = jax.lax.stop_gradient(tgt_q)

      # Compute the current Q estimates
      cur_q1, cur_q2 = self.critic.apply(
        params['critic'], observations, actions
      )

      # Critic loss
      critic1_loss = mse_loss(cur_q1, tgt_q)
      critic2_loss = mse_loss(cur_q2, tgt_q)
      critic_loss = critic1_loss + critic2_loss

      # Compute the actor loss
      new_actions = self.actor.apply(params['actor'], rng, observations)
      q = self.critic.apply(
        params['critic'], observations, new_actions, method=self.critic.q1
      )
      q_abs_mean = jax.lax.stop_gradient(jnp.abs(q).mean())
      lmbda = self.config.alpha / q_abs_mean
      bc_loss = mse_loss(new_actions, actions)
      actor_loss = -lmbda * q.mean() + bc_loss

      loss_collection = {'actor': actor_loss, 'critic': critic_loss}
      return tuple(loss_collection[key] for key in self.model_keys), locals()

    # Calculate losses and grads
    params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_values), grads = value_and_multi_grad(
      loss_fn, len(self.model_keys), has_aux=True
    )(params, tgt_params, rng)

    # Update critic train states
    train_states['critic'] = train_states['critic'].apply_gradients(
      grads=grads[1]['critic']
    )

    # Update actor train states if necessary
    if policy_update:
      train_states['actor'] = train_states['actor'].apply_gradients(
        grads=grads[0]['actor']
      )

      # Update target parameters
      tgt_params['actor'] = update_target_network(
        train_states['actor'].params, tgt_params['actor'], self.config.tau
      )
      tgt_params['critic'] = update_target_network(
        train_states['critic'].params, tgt_params['critic'], self.config.tau
      )

    metrics = dict(
      actor_loss=aux_values['actor_loss'],
      critic_loss=aux_values['critic_loss'],
      critic1_loss=aux_values['critic1_loss'],
      critic2_loss=aux_values['critic2_loss'],
      bc_loss=aux_values['bc_loss'],
      cur_q1=aux_values['cur_q1'].mean(),
      cur_q2=aux_values['cur_q2'].mean(),
      tgt_q1=aux_values['tgt_q1'].mean(),
      tgt_q2=aux_values['tgt_q2'].mean(),
      tgt_q=aux_values['tgt_q'].mean(),
      lmbda=aux_values['lmbda'],
      q_abs_mean=aux_values['q_abs_mean'],
    )

    return train_states, tgt_params, metrics

  def train(self, batch):
    self._total_steps += 1
    self._train_states, self._tgt_params, metrics = self._train_step(
      self._train_states,
      self._tgt_params,
      next_rng(),
      batch,
      self._total_steps % self.config.policy_freq == 0,
    )
    return metrics

  @property
  def model_keys(self):
    return self._model_keys

  @property
  def train_states(self):
    return self._train_states

  @property
  def train_params(self):
    return {key: self.train_states[key].params for key in self.model_keys}

  @property
  def total_steps(self):
    return self._total_steps
