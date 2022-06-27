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
    config.use_automatic_entropy_tuning = True
    config.backup_entropy = False
    config.target_entropy = 0.0
    config.encoder_lr = 3e-4
    config.policy_lr = 3e-4
    config.qf_lr = 3e-4
    config.optimizer_type = 'adam'
    config.soft_target_update_rate = 5e-3
    config.sample_actions = 4
    config.crr_fn = 'exp' # exp or indicator
    config.crr_beta = 1.0
    config.crr_ratio_upper_bound = 20.0
    config.nstep = 1

    if updates is not None:
      config.update(ConfigDict(updates).copy_and_resolve_references())
    return config

  def __init__(self, config, encoder, policy, qf):
    self.config = self.get_default_config(config)
    self.encoder = encoder
    self.policy = policy
    self.qf = qf
    self.observation_dim = policy.observation_dim
    self.action_dim = policy.action_dim

    self._train_states = {}

    optimizer_class = {
      'adam': optax.adam,
      'sgd': optax.sgd,
    }[self.config.optimizer_type]

    encoder_params = self.encoder.init(
      next_rng(), jnp.zeros((10, self.policy.observation_dim))
    )
    self._train_states['encoder'] = TrainState.create(
      params=encoder_params,
      tx=optimizer_class(self.config.encoder_lr),
      apply_fn=None
    )

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

    model_keys = ['policy', 'qf', 'encoder']

    self._model_keys = tuple(model_keys)
    self._total_steps = 0

  def train(self, batch, bc=False):
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
      embedding = self.encoder.apply(
        train_params['encoder'], observations
      )
      next_embedding = self.encoder.apply(
        train_params['encoder'], next_observations
      )
 
      rng, split_rng = jax.random.split(rng)
      _, log_pi = self.policy.apply(
        train_params['policy'], split_rng, embedding 
     )

      """ Q function loss """
      q1_pred = self.qf.apply(train_params['qf'], embedding, actions)

      rng, split_rng = jax.random.split(rng)
      new_next_actions, next_log_pi = self.policy.apply(
        train_params['policy'], split_rng, next_embedding 
      )
      target_q_values = self.qf.apply(
        target_qf_params['qf'], next_embedding, new_next_actions
      )

      q_target = jax.lax.stop_gradient(
        rewards + (1. - dones) * self.config.discount * target_q_values
      )
      qf_loss = mse_loss(q1_pred, q_target)

      loss_collection['qf'] = qf_loss

      rng, split_rng = jax.random.split(rng)
      replicated_obs = jnp.broadcast_to(
        embedding, (self.config.sample_actions,) + embedding.shape
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
        embedding,
        actions,
        method=self.policy.log_prob
      )
      policy_loss = -jnp.mean(log_prob * coef)
      loss_collection['policy'] = policy_loss
      loss_collection['encoder'] = (
        loss_collection['policy'] +
        loss_collection['qf']
      )

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
