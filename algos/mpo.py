from copy import deepcopy
from email import policy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from ml_collections import ConfigDict

from algos.model import Scalar, update_target_network
from utilities.jax_utils import mse_loss, next_rng, value_and_multi_grad
from algos.mpo_loss import MPO as mpo_loss


class MPO(object):

  @staticmethod
  def get_default_config(updates=None):
    config = ConfigDict()
    config.nstep = 1
    config.discount = 0.99
    config.alpha_multiplier = 1.0
    config.use_automatic_entropy_tuning = True
    config.backup_entropy = False
    config.target_entropy = 0.0
    config.encoder_lr = 3e-4
    config.policy_lr = 3e-4
    config.qf_lr = 3e-4
    config.mpo_lr = 3e-4
    config.optimizer_type = 'adam'
    config.soft_target_update_rate = 5e-3
    config.bc_mode = 'mse'  # 'mle'
    config.bc_weight = 0.
    config.res_hidden_size = 1024
    config.encoder_blocks = 1
    config.head_blocks = 1
    config.num_samples = 40
    config.target_policy_update_period = 100
    config.target_critic_update_period = 100
    config.clipping = True
    config.double_q = False
    config.decoupled_q = False
    config.init_log_temperature = 100.0
    config.init_log_alpha_mean = 10.0
    config.init_log_alpha_stddev = 1000.0
    config.epsilon = 1e-1
    config.epsilon_penalty = 1e-3
    config.epsilon_mean = 2.5e-3
    config.epsilon_stddev = 1e-6

    if updates is not None:
      config.update(ConfigDict(updates).copy_and_resolve_references())
    return config

  def __init__(self, config, encoder, policy, qf):
    self.config = self.get_default_config(config)
    self.policy = policy
    self.qf = qf
    self.encoder = encoder
    self.observation_dim = policy.input_size
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
      tx=optax.chain(
        optax.clip_by_global_norm(40.0),
        optimizer_class(self.config.encoder_lr),
      ),
      apply_fn=None
    )

    policy_params = self.policy.init(
      next_rng(), next_rng(), jnp.zeros((10, self.observation_dim))
    )
    self._train_states['policy'] = TrainState.create(
      params=policy_params,
      tx=optax.chain(
        optax.clip_by_global_norm(40.0),
        optimizer_class(self.config.policy_lr),
      ),
      apply_fn=None
    )

    qf_params = self.qf.init(
      next_rng(), jnp.zeros((10, self.observation_dim)),
      jnp.zeros((10, self.action_dim))
    )

    self._train_states['qf'] = TrainState.create(
      params=qf_params,
      tx=optax.chain(
        optax.clip_by_global_norm(40.0),
        optimizer_class(self.config.qf_lr),
      ),
      apply_fn=None,
    )
 
    if self.config.double_q:
      qf2_params = self.qf.init(
        next_rng(), jnp.zeros((10, self.observation_dim)),
        jnp.zeros((10, self.action_dim))
      )
      self._train_states['qf2'] = TrainState.create(
        params=qf2_params,
        tx=optax.chain(
          optax.clip_by_global_norm(40.0),
          optimizer_class(self.config.qf_lr),
        ),
        apply_fn=None,
      )
   
    self._mpo_loss = mpo_loss(
      epsilon=self.config.epsilon,
      epsilon_penalty=self.config.epsilon_penalty,
      epsilon_mean=self.config.epsilon_mean,
      epsilon_stddev=self.config.epsilon_stddev,
      init_log_temperature=self.config.init_log_temperature,
      init_log_alpha_mean=self.config.init_log_alpha_mean,
      init_log_alpha_stddev=self.config.init_log_alpha_stddev,
    )
    mpo_params = self._mpo_loss.init_params(
      self.policy.action_dim,
    )

    self._train_states['mpo'] = TrainState.create(
      params=mpo_params,
      tx=optax.chain(
        optimizer_class(self.config.mpo_lr),
      ),
      apply_fn=None
    )

    model_keys = ['policy', 'qf', 'encoder', 'mpo']
    self._total_steps = 0

    self._target_params = {
      'qf': deepcopy(qf_params),
      'policy': deepcopy(policy_params),
      'encoder': deepcopy(encoder_params),
      'mpo': deepcopy(mpo_params)
    }
    if self.config.double_q:
      model_keys.append('qf2')
      self._target_params['qf2'] = deepcopy(qf2_params)

    self._model_keys = tuple(model_keys)

  def train(self, batch, bc=False):
    self._total_steps += 1
    self._train_states, self._target_params, metrics = self._train_step(
      self._train_states, self._target_params, next_rng(), batch, bc
    )
    return metrics

  @partial(jax.jit, static_argnames=('self', 'bc'))
  def _train_step(self, train_states, target_params, rng, batch, bc=False):

    def loss_fn(train_params, rng):
      observations = batch['observations']
      actions = batch['actions']
      rewards = batch['rewards']
      next_observations = batch['next_observations']
      dones = batch['dones']

      loss_collection = {}

      rng, split_rng = jax.random.split(rng)
      embedding = self.encoder.apply(
        train_params['encoder'], observations
      )
      next_embedding = self.encoder.apply(
        target_params['encoder'], next_observations
      )
      next_embedding = jax.lax.stop_gradient(next_embedding)

      online_action_dist = self.policy.apply(
        train_params['policy'],
        next_embedding,
        method=self.policy.get_tfd_dist,
      )

      target_action_dist = self.policy.apply(
        target_params['policy'],
        next_embedding,
        method=self.policy.get_tfd_dist
      )
      rng, split_rng = jax.random.split(rng)
      sampled_actions = target_action_dist.sample(
        self.config.num_samples,
        seed=split_rng
      )

      sampled_q_t = self.qf.apply(
        target_params['qf'],
        jnp.repeat(
          jnp.expand_dims(next_embedding, axis=0),
          self.config.num_samples, axis=0
        ),
        sampled_actions, 
      )
      q_t = jnp.mean(sampled_q_t, axis=0)

      if self.config.double_q:
        sampled_q_t_2 = self.qf.apply(
          target_params['qf2'],
          jnp.repeat(
            jnp.expand_dims(next_embedding, axis=0),
            self.config.num_samples, axis=0
          ),
          sampled_actions, 
        )
        q_t = jnp.mean(
          jnp.minimum(sampled_q_t, sampled_q_t_2),
          axis=0
        )

      q_tm1 = self.qf.apply(
        train_params['qf'],
        embedding,
        actions
      )

      discount = self.config.discount**self.config.nstep
      td_target = jax.lax.stop_gradient(
        rewards + (1. - dones) * discount * q_t
      )
      qf_loss = mse_loss(q_tm1, td_target)
      loss_collection['qf'] = qf_loss

      if self.config.double_q:
        q_tm1_2 = self.qf.apply(
          train_params['qf2'],
          embedding,
          actions,
        )
        qf_loss_2 = mse_loss(q_tm1_2, td_target)
        loss_collection['qf2'] = qf_loss_2

      rng, split_rng = jax.random.split(rng)

      policy_loss, stats = self._mpo_loss(
        train_params['mpo'],
        online_action_dist,
        target_action_dist,
        sampled_actions,
        sampled_q_t,
      )
      policy_loss = policy_loss[0]

      loss_collection['policy'] = policy_loss
      loss_collection['encoder'] = (
        loss_collection['policy'] +
        loss_collection['qf']
      )

      mpo_loss = stats.loss_alpha + stats.loss_temperature
      loss_collection['mpo'] = mpo_loss
      return tuple(loss_collection[key] for key in self.model_keys), locals()

    train_params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_values), grads = value_and_multi_grad(
      loss_fn, len(self.model_keys), has_aux=True
    )(train_params, rng)

    mpo_stats = aux_values['stats']

    new_train_states = {
      key: train_states[key].apply_gradients(grads=grads[i][key])
      for i, key in enumerate(self.model_keys)
    }

    new_target_params = dict()
    for k, v in new_train_states.items():
      new_target_params[k] = update_target_network(
        v.params, target_params[k], self.config.soft_target_update_rate
      )

    metrics = dict(
      policy_loss=aux_values['policy_loss'],
      mpo_loss=aux_values['mpo_loss'],
      qf_loss=aux_values['qf_loss'],
      average_qf=aux_values['q_tm1'].mean(),
      average_target_q=aux_values['q_t'].mean(),
      dual_alpha_mean=mpo_stats.dual_alpha_mean,
      dual_alpha_stddev=mpo_stats.dual_alpha_stddev,
      dual_temperature=mpo_stats.dual_temperature,
      loss_alpha=mpo_stats.loss_alpha,
      loss_temperature=mpo_stats.loss_temperature,
    )

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
