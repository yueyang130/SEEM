from copy import deepcopy
from functools import partial
from logging import logProcesses

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from ml_collections import ConfigDict

from algos.model import Scalar, update_target_network
from utilities.jax_utils import mse_loss, next_rng, value_and_multi_grad


class CRR(object):

  @staticmethod
  def get_default_config(updates=None):
    config = ConfigDict()
    config.discount = 0.99
    config.alpha_multiplier = 1.0
    config.use_automatic_entropy_tuning = True
    config.backup_entropy = False
    config.target_entropy = 0.0
    config.policy_lr = 3e-4
    config.qf_lr = 3e-4
    config.optimizer_type = 'adam'
    config.soft_target_update_rate = 5e-3
    config.sample_actions = 4
    config.crr_fn = 'exp' # exp or indicator
    config.crr_beta = 1.0
    config.crr_ratio_upper_bound = 20.0

    if updates is not None:
      config.update(ConfigDict(updates).copy_and_resolve_references())
    return config

  def __init__(self, config, policy, qf):
    self.config = self.get_default_config(config)
    self.policy = policy
    self.qf = qf
    self.observation_dim = policy.observation_dim
    self.action_dim = policy.action_dim

    self._train_states = {}

    optimizer_class = {
      'adam': optax.adam,
      'sgd': optax.sgd,
    }[self.config.optimizer_type]

    policy_params = self.policy.init(
      next_rng(), next_rng(), jnp.zeros((10, self.observation_dim))
    )
    self._train_states['policy'] = TrainState.create(
      params=policy_params,
      tx=optimizer_class(self.config.policy_lr),
      apply_fn=None
    )

    qf_params = self.qf.init(
      next_rng(), jnp.zeros((10, self.observation_dim)),
      jnp.zeros((10, self.action_dim))
    )
    self._train_states['qf'] = TrainState.create(
      params=qf_params,
      tx=optimizer_class(self.config.qf_lr),
      apply_fn=None,
    )
    self._target_qf_params = deepcopy({'qf': qf_params})

    model_keys = ['policy', 'qf']

    if self.config.use_automatic_entropy_tuning:
      self.log_alpha = Scalar(0.0)
      self._train_states['log_alpha'] = TrainState.create(
        params=self.log_alpha.init(next_rng()),
        tx=optimizer_class(self.config.policy_lr),
        apply_fn=None
      )
      model_keys.append('log_alpha')

    self._model_keys = tuple(model_keys)
    self._total_steps = 0

  def train(self, batch):
    self._total_steps += 1
    self._train_states, self._target_qf_params, metrics = self._train_step(
      self._train_states, self._target_qf_params, next_rng(), batch
    )
    return metrics
  
  @partial(jax.jit, static_argnames='self')
  def _train_step(self, train_states, target_qf_params, rng, batch):

    def loss_fn(train_params, rng):
      observations = batch['observations']
      actions = batch['actions']
      rewards = batch['rewards']
      next_observations = batch['next_observations']
      dones = batch['dones']

      loss_collection = {}

      rng, split_rng = jax.random.split(rng)
      _, log_pi = self.policy.apply(
        train_params['policy'], split_rng, observations
     )

      if self.config.use_automatic_entropy_tuning:
        alpha_loss = -self.log_alpha.apply(train_params['log_alpha']) * (
          log_pi + self.config.target_entropy
        ).mean()
        loss_collection['log_alpha'] = alpha_loss
        alpha = jax.nn.softplus(
          self.log_alpha.apply(train_params['log_alpha'])
        ) * self.config.alpha_multiplier
      else:
        alpha_loss = 0.0
        alpha = self.config.alpha_multiplier

      """ Q function loss """
      q1_pred = self.qf.apply(train_params['qf'], observations, actions)

      rng, split_rng = jax.random.split(rng)
      new_next_actions, next_log_pi = self.policy.apply(
        train_params['policy'], split_rng, next_observations
      )
      target_q_values = self.qf.apply(
        target_qf_params['qf'], next_observations, new_next_actions
      )

      if self.config.backup_entropy:
        target_q_values = target_q_values - alpha * next_log_pi

      q_target = jax.lax.stop_gradient(
        rewards + (1. - dones) * self.config.discount * target_q_values
      )
      qf_loss = mse_loss(q1_pred, q_target)

      loss_collection['qf'] = qf_loss

      rng, split_rng = jax.random.split(rng)
      replicated_obs = jnp.broadcast_to(
        observations, (self.config.sample_actions,) + observations.shape
      )
      vf_actions, _ = self.policy.apply(
        train_params['policy'], split_rng, replicated_obs
      )
      v = jnp.mean(
        self.qf.apply(
          train_params['qf'], replicated_obs, vf_actions
        )
      )

      adv = q1_pred - v

      if self.config.crr_fn == 'exp':
        coef = jnp.minimum(self.config.crr_ratio_upper_bound, jnp.exp(adv / self.config.crr_beta))
      else:
        coef = jnp.heaviside(adv, 0)
      
      coef = jax.lax.stop_gradient(coef)
      log_prob = self.policy.apply(
        train_params['policy'],
        observations,
        actions,
        method=self.policy.log_prob
      )
      policy_loss = -jnp.mean(log_prob * coef)
      loss_collection['policy'] = policy_loss

      return tuple(loss_collection[key] for key in self.model_keys), locals()

    train_params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_values), grads = value_and_multi_grad(
      loss_fn, len(self.model_keys), has_aux=True
    )(train_params, rng)

    new_train_states = {
      key: train_states[key].apply_gradients(grads=grads[i][key])
      for i, key in enumerate(self.model_keys)
    }
    new_target_qf_params = {}
    new_target_qf_params['qf'] = update_target_network(
      new_train_states['qf'].params, target_qf_params['qf'],
      self.config.soft_target_update_rate
    )

    metrics = dict(
      log_pi=aux_values['log_pi'].mean(),
      policy_loss=aux_values['policy_loss'],
      qf_loss=aux_values['qf_loss'],
      alpha_loss=aux_values['alpha_loss'],
      alpha=aux_values['alpha'],
      average_qf=aux_values['q1_pred'].mean(),
      average_target_q=aux_values['target_q_values'].mean(),
    )
    return new_train_states, new_target_qf_params, metrics

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
