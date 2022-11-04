from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from ml_collections import ConfigDict

from algos.model import Scalar, update_target_network
from core.core_api import Algo
from utilities.jax_utils import mse_loss, next_rng, value_and_multi_grad
from utilities.utils import prefix_metrics


class IQL(Algo):

  @staticmethod
  def get_default_config(updates=None):
    config = ConfigDict()
    config.nstep = 1
    config.discount = 0.99
    config.alpha_multiplier = 1.0
    config.use_automatic_entropy_tuning = True
    config.backup_entropy = False
    config.target_entropy = 0.0
    config.policy_lr = 1e-4
    config.qf_lr = 3e-4
    config.vf_lr = 3e-4
    config.optimizer_type = 'adam'
    config.soft_target_update_rate = 5e-3
    config.bc_mode = 'mse'  # 'mle'
    config.bc_weight = 0.
    config.res_hidden_size = 1024
    config.head_blocks = 1
    config.expectile = 0.7
    config.awr_temperature = 3.0
    config.loss_type = 'expectile'

    if updates is not None:
      config.update(ConfigDict(updates).copy_and_resolve_references())
    return config

  def __init__(self, config, policy, qf, vf):
    self.config = self.get_default_config(config)
    self.policy = policy
    self.qf = qf
    self.vf = vf
    self.observation_dim = policy.input_size
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

    qf1_params = self.qf.init(
      next_rng(), jnp.zeros((10, self.observation_dim)),
      jnp.zeros((10, self.action_dim))
    )
    self._train_states['qf1'] = TrainState.create(
      params=qf1_params,
      tx=optimizer_class(self.config.qf_lr),
      apply_fn=None,
    )
    qf2_params = self.qf.init(
      next_rng(), jnp.zeros((10, self.observation_dim)),
      jnp.zeros((10, self.action_dim))
    )
    self._train_states['qf2'] = TrainState.create(
      params=qf2_params,
      tx=optimizer_class(self.config.qf_lr),
      apply_fn=None,
    )

    vf_params = self.vf.init(next_rng(), jnp.zeros((10, self.observation_dim)))
    self._train_states['vf'] = TrainState.create(
      params=vf_params, tx=optimizer_class(self.config.vf_lr), apply_fn=None
    )

    self._target_qf_params = deepcopy({'qf1': qf1_params, 'qf2': qf2_params})
    model_keys = ['policy', 'qf1', 'qf2', 'vf']

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

  @partial(jax.jit, static_argnames=('self'))
  def _train_step(self, train_states, target_qf_params, rng, batch):
    observations = batch['observations']
    actions = batch['actions']
    rewards = batch['rewards']
    next_observations = batch['next_observations']
    dones = batch['dones']

    def value_loss(train_params):
      q1 = self.qf.apply(target_qf_params['qf1'], observations, actions)
      q2 = self.qf.apply(target_qf_params['qf2'], observations, actions)
      q_pred = jax.lax.stop_gradient(jnp.minimum(q1, q2))
      v_pred = self.vf.apply(train_params['vf'], observations)
      diff = q_pred - v_pred
      expectile_weight = jnp.where(
        diff > 0,
        self.config.expectile,
        1 - self.config.expectile,
      )

      if self.config.loss_type == 'expectile':
        expectile_loss = (expectile_weight * (diff**2)).mean()
      elif self.config.loss_type == 'quantile':
        expectile_loss = (expectile_weight * (jnp.abs(diff))).mean()
      else:
        raise NotImplementedError

      return (expectile_loss,), locals()

    def awr_loss(train_params):
      v_pred = self.vf.apply(train_params['vf'], observations)
      q1 = self.qf.apply(target_qf_params['qf1'], observations, actions)
      q2 = self.qf.apply(target_qf_params['qf2'], observations, actions)
      q_pred = jax.lax.stop_gradient(jnp.minimum(q1, q2))
      exp_a = jnp.exp((q_pred - v_pred) * self.config.awr_temperature)
      exp_a = jnp.minimum(exp_a, 100.0)
      log_probs = self.policy.apply(
        train_params['policy'],
        observations,
        actions,
        method=self.policy.log_prob
      )
      awr_loss = -(exp_a * log_probs).mean()

      return (awr_loss,), locals()

    def critic_loss(train_params):
      next_v = self.vf.apply(train_params['vf'], next_observations)

      discount = self.config.discount**self.config.nstep
      td_target = jax.lax.stop_gradient(
        rewards + (1 - dones) * discount * next_v
      )

      q1_pred = self.qf.apply(train_params['qf1'], observations, actions)
      q2_pred = self.qf.apply(train_params['qf2'], observations, actions)
      qf1_loss = mse_loss(q1_pred, td_target)
      qf2_loss = mse_loss(q2_pred, td_target)

      return (qf1_loss, qf2_loss), locals()

    train_params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_value), value_grads = value_and_multi_grad(
      value_loss, 1, has_aux=True
    )(
      train_params
    )
    train_states['vf'] = train_states['vf'].apply_gradients(
      grads=value_grads[0]['vf']
    )

    (_, aux_policy), policy_grads = value_and_multi_grad(
      awr_loss, 1, has_aux=True
    )(
      train_params
    )
    train_states['policy'] = train_states['policy'].apply_gradients(
      grads=policy_grads[0]['policy']
    )

    (_, aux_qf), qf_grads = value_and_multi_grad(
      critic_loss, 2, has_aux=True
    )(
      train_params
    )

    for i, k in enumerate(['qf1', 'qf2']):
      train_states[k] = train_states[k].apply_gradients(grads=qf_grads[i][k])

    new_target_qf_params = {}
    new_target_qf_params['qf1'] = update_target_network(
      train_states['qf1'].params, target_qf_params['qf1'],
      self.config.soft_target_update_rate
    )
    new_target_qf_params['qf2'] = update_target_network(
      train_states['qf2'].params, target_qf_params['qf2'],
      self.config.soft_target_update_rate
    )

    metrics = dict(
      v_pred=aux_value['v_pred'].mean(),
      q1_pred=aux_value['q1'].mean(),
      q2_pred=aux_value['q2'].mean(),
      expectile_weight=aux_value['expectile_weight'].mean(),
      expectile_loss=aux_value['expectile_loss'],
      awr_exp_a=aux_policy['exp_a'].mean(),
      awr_log_prob=aux_policy['log_probs'].mean(),
      awr_loss=aux_policy['awr_loss'],
      qf1_loss=aux_qf['qf1_loss'],
      qf2_loss=aux_qf['qf2_loss'],
      td_target=aux_qf['td_target'].mean(),
    )

    return train_states, new_target_qf_params, metrics

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
