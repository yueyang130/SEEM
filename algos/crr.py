from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from ml_collections import ConfigDict

from algos.model import update_target_network
from utilities.jax_utils import mse_loss, next_rng, value_and_multi_grad


class CRR(object):

  @staticmethod
  def get_default_config(updates=None):
    config = ConfigDict()
    config.discount = 0.99
    config.use_automatic_entropy_tuning = True
    config.backup_entropy = False
    config.target_entropy = 0.0
    config.policy_lr = 1e-4
    config.qf_lr = 3e-4
    config.optimizer_type = 'adam'
    config.soft_target_update_rate = 5e-3
    config.sample_actions = 50
    config.crr_fn = 'exp' # exp or indicator
    config.crr_beta = 1.0
    config.crr_ratio_upper_bound = 20.0
    config.nstep = 1
    config.double_q = True
    config.avg_fn = 'mean' # or max
    config.merge_all = True
    config.q_weight_method = 'min'
    config.use_expectile = True
    config.exp_tau = 0.7

    if updates is not None:
      config.update(ConfigDict(updates).copy_and_resolve_references())
    return config

  def __init__(self, config, policy, qf, decoupled_q=False):
    self.config = self.get_default_config(config)
    self.decoupled_q = decoupled_q
    self.policy = policy
    self.qf = qf
    self.observations_dim = policy.input_size
    self.action_dim = policy.action_dim

    self._train_states = {}

    optimizer_class = {
      'adam': optax.adam,
      'sgd': optax.sgd,
    }[self.config.optimizer_type]

    policy_params = self.policy.init(
      next_rng(), next_rng(), jnp.zeros((10, self.observations_dim))
    )
    self._train_states['policy'] = TrainState.create(
      params=policy_params,
      tx=optimizer_class(self.config.policy_lr),
      apply_fn=None
    )

    qf_params = self.qf.init(
      next_rng(), jnp.zeros((10, self.observations_dim)),
      jnp.zeros((10, self.action_dim))
    )
    self._train_states['qf'] = TrainState.create(
      params=qf_params,
      tx=optimizer_class(self.config.qf_lr),
      apply_fn=None,
    )
    target_dict = {'qf': qf_params}
    model_keys = ['policy', 'qf']

    if self.config.double_q:
      qf2_params = self.qf.init(
        next_rng(), jnp.zeros((10, self.observations_dim)),
        jnp.zeros((10, self.action_dim))
      )
      self._train_states['qf2'] = TrainState.create(
        params=qf2_params, tx=optimizer_class(self.config.qf_lr), apply_fn=None
      )
      target_dict['qf2'] = qf2_params
      model_keys.append('qf2')

    target_dict['policy'] = policy_params
    self._target_params = deepcopy(target_dict)

    self._model_keys = tuple(model_keys)
    self._total_steps = 0

  def train(self, batch, bc=False):
    self._total_steps += 1
    self._train_states, self._target_params, metrics = self._train_step(
      self._train_states, self._target_params, next_rng(), batch
    )
    return metrics
  
  @partial(jax.jit, static_argnames='self')
  def _train_step(self, train_states, target_params, rng, batch):

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

      """ Q function loss """
      q1_pred = self.qf.apply(train_params['qf'], observations, actions)
      if self.config.double_q:
        q2_pred = self.qf.apply(train_params['qf2'], observations, actions)

      rng, split_rng = jax.random.split(rng)
      new_next_actions, next_log_pi = self.policy.apply(
        target_params['policy'], split_rng, next_observations 
      )
      target_q_values = self.qf.apply(
        target_params['qf'], next_observations, new_next_actions
      )
      if self.config.double_q:
        target_q_values = jnp.minimum(
          target_q_values,
          self.qf.apply(target_params['qf2'], next_observations,
            new_next_actions)
        )

      q_target = jax.lax.stop_gradient(
        rewards + (1. - dones) * self.config.discount * target_q_values
      )
      qf_loss = mse_loss(q1_pred, q_target)

      loss_collection['qf'] = qf_loss * 0.5

      if self.config.double_q:
        qf2_loss = mse_loss(q2_pred, q_target)
        loss_collection['qf2'] = qf2_loss * 0.5
      
      if self.config.use_expectile:
        diff1 = q1_pred - q_target
        exp_w1 = jnp.where(
          diff1 > 0,
          self.config.exp_tau,
          1 - self.config.exp_tau
        )
        loss_collection['qf'] = (exp_w1 * (diff1 ** 2)).mean()

        if self.config.double_q:
          diff2 = q2_pred - q_target
          exp_w2 = jnp.where(
            diff2 > 0,
            self.config.exp_tau,
            1 - self.config.exp_tau
          )

          loss_collection['qf2'] = (
            exp_w2 * (diff2 ** 2)
          ).mean()

      rng, split_rng = jax.random.split(rng)
      replicated_obs = jnp.broadcast_to(
        observations, (self.config.sample_actions,) + observations.shape
      )
      vf_actions, _ = self.policy.apply(
        train_params['policy'], split_rng, replicated_obs
      )

      v = self.qf.apply(train_params['qf'], replicated_obs, vf_actions)
      q_pred = q1_pred
      if self.config.double_q:
        v2 = self.qf.apply(train_params['qf2'], replicated_obs, vf_actions)
        if self.config.q_weight_method == 'min':
          v = jnp.minimum(v, v2)
        elif self.config.q_weight_method == 'avg':
          v = (v + v2) / 2
        else:
          raise NotImplementedError
        q_pred = jnp.minimum(q1_pred, q2_pred)

      if self.config.merge_all:
        adv = q_pred - getattr(jnp, self.config.avg_fn)(v)
      else:
        adv = q_pred - getattr(jnp, self.config.avg_fn)(v, axis=0)

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
    new_target_params = {}
    for k in target_params.keys():
      new_target_params[k] = update_target_network(new_train_states[k].params,
          target_params[k], self.config.soft_target_update_rate)

    metrics = dict(
      log_pi=aux_values['log_pi'].mean(),
      policy_loss=aux_values['policy_loss'],
      qf_loss=aux_values['qf_loss'],
      average_qf=aux_values['q1_pred'].mean(),
      average_target_q=aux_values['target_q_values'].mean(),
      adv=aux_values['adv'].mean(),
      coef=aux_values['coef'].mean(),
      v=aux_values['v'].mean()
    )

    if self.config.double_q:
      q2_dict = dict(
        q2_pred=aux_values['q2_pred'].mean(),
        qf2_loss=aux_values['qf2_loss']
        )
      metrics.update(q2_dict)

    return new_train_states, new_target_params, metrics

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
