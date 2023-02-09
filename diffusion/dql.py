from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import optax

from flax.training.train_state import TrainState
from ml_collections import ConfigDict
from core.core_api import Algo
from utilities.jax_utils import mse_loss, next_rng, value_and_multi_grad

from diffusion.diffusion import GaussianDiffusion
import distrax


def update_target_network(main_params, target_params, tau):
  return jax.tree_multimap(
    lambda x, y: tau * x + (1.0 - tau) * y, main_params, target_params
  )


class DiffusionQL(Algo):

  @staticmethod
  def get_default_config(updates=None):
    cfg = ConfigDict()
    cfg.nstep = 1
    cfg.discount = 0.99
    cfg.tau = 0.005
    cfg.policy_tgt_freq = 5
    cfg.num_timesteps = 100
    cfg.schedule_name = 'linear'
    cfg.time_embed_size = 16
    cfg.alpha = 2.  # NOTE 0.25 in diffusion rl but 2.5 in td3
    cfg.use_pred_astart = True
    cfg.max_q_backup = False
    cfg.max_q_backup_topk = 1
    cfg.max_q_backup_samples = 10

    # learning related
    cfg.lr = 3e-4
    cfg.diff_coef = 1.0
    cfg.guide_coef = 1.0
    cfg.lr_decay = False
    cfg.lr_decay_steps = 1000000
    cfg.max_grad_norm = 0.
    cfg.weight_decay = 0.

    cfg.loss_type = 'TD3'

    # CRR-related hps
    cfg.use_expectile = False
    cfg.exp_tau = 0.7
    cfg.sample_actions = 20
    cfg.crr_ratio_upper_bound = 20
    cfg.crr_beta = 1.0
    cfg.crr_weight_mode = 'mle'
    cfg.crr_fixed_std = True
    cfg.crr_multi_sample_mse = False
    cfg.crr_avg_fn = 'mean'
    cfg.crr_fn = 'exp'
    cfg.crr_adv_norm = False

    # for dpm-solver
    cfg.dpm_steps = 15
    cfg.dpm_t_end = 0.001

    # useless
    cfg.target_entropy = -1
    if updates is not None:
      cfg.update(ConfigDict(updates).copy_and_resolve_references())
    return cfg

  def __init__(self, cfg, policy, qf, policy_dist):
    self.config = self.get_default_config(cfg)
    self.policy = policy
    self.qf = qf
    self.policy_dist = policy_dist
    self.observation_dim = policy.observation_dim
    self.action_dim = policy.action_dim
    self.max_action = policy.max_action
    self.diffusion: GaussianDiffusion = self.policy.diffusion

    self._total_steps = 0
    self._train_states = {}

    policy_params = self.policy.init(
      next_rng(),
      next_rng(),
      jnp.zeros((10, self.observation_dim)),
    )

    def get_lr(lr_decay=False):
      if lr_decay is True:
        return optax.cosine_decay_schedule(
          self.config.lr, decay_steps=self.config.lr_decay_steps
        )
      else:
        return self.config.lr

    def get_optimizer(lr_decay=False, weight_decay=cfg.weight_decay):
      if self.config.max_grad_norm > 0:
        opt = optax.chain(
          optax.clip_by_global_norm(self.config.max_grad_norm),
          optax.adamw(get_lr(lr_decay), weight_decay=weight_decay),
        )
      else:
        opt = optax.adamw(get_lr(), weight_decay=weight_decay)

      return opt

    self._train_states['policy'] = TrainState.create(
      params=policy_params,
      tx=get_optimizer(self.config.lr_decay, weight_decay=0.0),
      apply_fn=None
    )

    policy_dist_params = self.policy_dist.init(
      next_rng(), jnp.zeros((10, self.action_dim))
    )
    self._train_states['policy_dist'] = TrainState.create(
      params=policy_dist_params, tx=get_optimizer(), apply_fn=None
    )

    qf1_params = self.qf.init(
      next_rng(),
      jnp.zeros((10, self.observation_dim)),
      jnp.zeros((10, self.action_dim)),
    )
    qf2_params = self.qf.init(
      next_rng(),
      jnp.zeros((10, self.observation_dim)),
      jnp.zeros((10, self.action_dim)),
    )

    self._train_states['qf1'] = TrainState.create(
      params=qf1_params, tx=get_optimizer(), apply_fn=None
    )
    self._train_states['qf2'] = TrainState.create(
      params=qf2_params, tx=get_optimizer(), apply_fn=None
    )
    self._tgt_params = deepcopy(
      {
        'policy': policy_params,
        'qf1': qf1_params,
        'qf2': qf2_params,
      }
    )
    self._model_keys = ('policy', 'qf1', 'qf2', 'policy_dist')

  def get_value_loss(self, batch):

    def value_loss_fn(params, tgt_params, rng):
      observations = batch['observations']
      actions = batch['actions']
      rewards = batch['rewards']
      next_observations = batch['next_observations']
      dones = batch['dones']

      # Compute the target Q values (without gradient)
      if self.config.max_q_backup:
        samples = self.config.max_q_backup_samples
        next_action = self.policy.apply(
          tgt_params['policy'], rng, next_observations, repeat=samples
        )
        next_action = jnp.clip(next_action, -self.max_action, self.max_action)
        next_obs_repeat = jnp.repeat(
          jnp.expand_dims(next_observations, axis=1), samples, axis=1
        )
        tgt_q1 = self.qf.apply(tgt_params['qf1'], next_obs_repeat, next_action)
        tgt_q2 = self.qf.apply(tgt_params['qf2'], next_obs_repeat, next_action)

        tk = self.config.max_q_backup_topk
        if tk == 1:
          tgt_q = jnp.minimum(tgt_q1.max(axis=-1), tgt_q2.max(axis=-1))
        else:
          batch_idx = jax.vmap(lambda x, i: x[i], 0)
          tgt_q1_max = batch_idx(tgt_q1, jnp.argsort(tgt_q1, axis=-1)[:, -tk])
          tgt_q2_max = batch_idx(tgt_q2, jnp.argsort(tgt_q2, axis=-1)[:, -tk])
          tgt_q = jnp.minimum(tgt_q1_max, tgt_q2_max)
      else:
        next_action = self.policy.apply(
          tgt_params['policy'], rng, next_observations
        )
        tgt_q1 = self.qf.apply(
          tgt_params['qf1'], next_observations, next_action
        )
        tgt_q2 = self.qf.apply(
          tgt_params['qf2'], next_observations, next_action
        )
        tgt_q = jnp.minimum(tgt_q1, tgt_q2)
      tgt_q = rewards + (1 - dones) * self.config.discount * tgt_q
      tgt_q = jax.lax.stop_gradient(tgt_q)

      # Compute the current Q estimates
      cur_q1 = self.qf.apply(params['qf1'], observations, actions)
      cur_q2 = self.qf.apply(params['qf2'], observations, actions)

      # qf loss
      qf1_loss = mse_loss(cur_q1, tgt_q)
      qf2_loss = mse_loss(cur_q2, tgt_q)

      if self.config.use_expectile:
        # diff1 = cur_q1 - tgt_q
        diff1 = tgt_q - cur_q1
        exp_w1 = jnp.where(
          diff1 > 0, self.config.exp_tau, 1 - self.config.exp_tau
        )
        qf1_loss = (exp_w1 * (diff1**2)).mean()
        # diff2 = cur_q2 - tgt_q
        diff2 = tgt_q - cur_q2
        exp_w2 = jnp.where(
          diff2 > 0, self.config.exp_tau, 1 - self.config.exp_tau
        )
        qf2_loss = (exp_w2 * (diff2**2)).mean()

      qf_loss = qf1_loss + qf2_loss
      return (qf1_loss, qf2_loss), locals()

    return value_loss_fn

  def get_diff_loss(self, batch):

    def diff_loss(params, rng):
      observations = batch['observations']
      actions = batch['actions']
      dones = batch['dones']

      # diffusion loss
      rng, split_rng = jax.random.split(rng)
      ts = jax.random.randint(
        split_rng, dones.shape, minval=0, maxval=self.diffusion.num_timesteps
      )
      rng, split_rng = jax.random.split(rng)
      terms = self.policy.apply(
        params["policy"],
        split_rng,
        observations,
        actions,
        ts,
        method=self.policy.loss,
      )
      diff_loss = terms["loss"].mean()

      if self.config.use_pred_astart:
        pred_astart = self.diffusion.p_mean_variance(
          terms["model_output"], terms["x_t"], ts
        )["pred_xstart"]
      else:
        rng, split_rng = jax.random.split(rng)
        pred_astart = self.policy.apply(
          params['policy'], split_rng, observations
        )
      terms["pred_astart"] = pred_astart
      return diff_loss, terms, ts, pred_astart

    return diff_loss

  @partial(jax.jit, static_argnames=('self', 'policy_tgt_update'))
  def _train_step(
    self, train_states, tgt_params, rng, batch, policy_tgt_update=False
  ):
    if self.config.loss_type not in ['TD3', 'CRR', 'Rainbow']:
      raise NotImplementedError

    return getattr(self, f"_train_step_{self.config.loss_type.lower()}"
                  )(train_states, tgt_params, rng, batch, policy_tgt_update)

  def _train_step_rainbow(
    self, train_states, tgt_params, rng, batch, policy_tgt_update=False
  ):
    value_loss_fn = self.get_value_loss(batch)
    diff_loss_fn = self.get_diff_loss(batch)

    def td3_loss_fn(params, tgt_params, rng, pred_astart):
      observations = batch['observations']

      # Calculate guide loss
      def fn(key):
        q = self.qf.apply(params[key], observations, pred_astart)
        lmbda = self.config.alpha / jax.lax.stop_gradient(jnp.abs(q).mean())
        policy_loss = -lmbda * q.mean()
        return lmbda, policy_loss

      lmbda, guide_loss = jax.lax.cond(
        jax.random.uniform(rng) > 0.5, partial(fn, 'qf1'), partial(fn, 'qf2')
      )
      return guide_loss

    def crr_loss_fn(params, tgt_params, rng, pred_astart):
      observations = batch['observations']
      actions = batch['actions']

      # Construct the policy distribution
      action_dist = self.policy_dist.apply(params['policy_dist'], pred_astart)
      if self.config.crr_fixed_std:
        action_dist = distrax.MultivariateNormalDiag(
          pred_astart, jnp.ones_like(pred_astart)
        )

      # Build action distribution
      replicated_obs = jnp.broadcast_to(
        observations, (self.config.sample_actions,) + observations.shape
      )
      rng, split_rng = jax.random.split(rng)
      if self.config.use_pred_astart:
        vf_actions = action_dist.sample(
          seed=split_rng, sample_shape=self.config.sample_actions
        )
      else:
        vf_actions = self.policy.apply(
          params['policy'], split_rng, replicated_obs
        )

      # Compute the current Q estimates
      cur_q1 = self.qf.apply(params['qf1'], observations, actions)
      cur_q2 = self.qf.apply(params['qf2'], observations, actions)

      # Compute values
      v1 = self.qf.apply(params['qf1'], replicated_obs, vf_actions)
      v2 = self.qf.apply(params['qf2'], replicated_obs, vf_actions)
      v = jnp.minimum(v1, v2)
      q_pred = jnp.minimum(cur_q1, cur_q2)
      avg_fn = getattr(jnp, self.config.crr_avg_fn)
      adv = q_pred - avg_fn(v, axis=0)
      if self.config.crr_fn == 'exp':
        lmbda = jnp.minimum(
          self.config.crr_ratio_upper_bound,
          jnp.exp(adv / self.config.crr_beta)
        )
        if self.config.crr_adv_norm:
          lmbda = jax.nn.softmax(adv / self.config.crr_beta)
      else:
        lmbda = jnp.heaviside(adv, 0)
      lmbda = jax.lax.stop_gradient(lmbda)
      if self.config.crr_weight_mode == 'elbo':
        log_prob = -terms['ts_weights'] * terms['mse']
      elif self.config.crr_weight_mode == 'mle':
        log_prob = action_dist.log_prob(actions)
      else:
        rng, split_rng = jax.random.split(rng)
        if not self.config.crr_multi_sample_mse:
          sampled_actions = action_dist.sample(seed=split_rng)
          log_prob = -((sampled_actions - actions)**2).mean(axis=-1)
        else:
          sampled_actions = action_dist.sample(
            seed=split_rng, sample_shape=self.config.sample_actions
          )
          log_prob = -(
            (sampled_actions - jnp.expand_dims(actions, axis=0))**2
          ).mean(axis=(0, -1))
      guide_loss = -jnp.mean(log_prob * lmbda)
      return guide_loss

    def policy_loss_fn(params, tgt_params, rng):

      rng, split_rng = jax.random.split(rng)
      diff_loss, _, _, pred_astart = diff_loss_fn(params, split_rng)

      td3_loss = td3_loss_fn(params, tgt_params, rng, pred_astart)
      crr_loss = crr_loss_fn(params, tgt_params, rng, pred_astart)

      policy_loss = diff_loss + self.config.guide_coef * (td3_loss + crr_loss)

      return (policy_loss,), locals()

    # Calculat q losses and grads
    params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_qf), grads_qf = value_and_multi_grad(
      value_loss_fn, 2, has_aux=True
    )(params, tgt_params, rng)

    # Calculat policy losses and grads
    params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_policy), grads_policy = value_and_multi_grad(
      policy_loss_fn, 1, has_aux=True
    )(params, tgt_params, rng)

    # Update qf train states
    train_states['qf1'] = train_states['qf1'].apply_gradients(
      grads=grads_qf[0]['qf1']
    )
    train_states['qf2'] = train_states['qf2'].apply_gradients(
      grads=grads_qf[1]['qf2']
    )

    # Update policy train states
    train_states['policy'] = train_states['policy'].apply_gradients(
      grads=grads_policy[0]['policy']
    )

    # Update target parameters
    if policy_tgt_update:
      tgt_params['policy'] = update_target_network(
        train_states['policy'].params, tgt_params['policy'], self.config.tau
      )
    tgt_params['qf1'] = update_target_network(
      train_states['qf1'].params, tgt_params['qf1'], self.config.tau
    )
    tgt_params['qf2'] = update_target_network(
      train_states['qf2'].params, tgt_params['qf2'], self.config.tau
    )

    metrics = dict(
      qf_loss=aux_qf['qf_loss'],
      qf1_loss=aux_qf['qf1_loss'],
      qf2_loss=aux_qf['qf2_loss'],
      cur_q1=aux_qf['cur_q1'].mean(),
      cur_q2=aux_qf['cur_q2'].mean(),
      tgt_q1=aux_qf['tgt_q1'].mean(),
      tgt_q2=aux_qf['tgt_q2'].mean(),
      tgt_q=aux_qf['tgt_q'].mean(),
      policy_loss=aux_policy['policy_loss'],
      direct_loss=aux_policy['td3_loss'],
      indirect_loss=aux_policy['crr_loss'],
      diff_loss=aux_policy['diff_loss'],
      # lmbda=aux_policy['lmbda'].mean(),
      qf1_grad_norm=optax.global_norm(grads_qf[0]['qf1']),
      qf2_grad_norm=optax.global_norm(grads_qf[1]['qf2']),
      policy_grad_norm=optax.global_norm(grads_policy[0]['policy']),
      qf1_weight_norm=optax.global_norm(train_states['qf1'].params),
      qf2_weight_norm=optax.global_norm(train_states['qf2'].params),
      policy_weight_norm=optax.global_norm(train_states['policy'].params),
    )

    return train_states, tgt_params, metrics

  def _train_step_td3(
    self, train_states, tgt_params, rng, batch, policy_tgt_update=False
  ):
    value_loss_fn = self.get_value_loss(batch)
    diff_loss_fn = self.get_diff_loss(batch)

    def policy_loss_fn(params, tgt_params, rng):
      observations = batch['observations']

      rng, split_rng = jax.random.split(rng)
      diff_loss, _, _, pred_astart = diff_loss_fn(params, split_rng)

      # Calculate guide loss
      def fn(key):
        q = self.qf.apply(params[key], observations, pred_astart)
        lmbda = self.config.alpha / jax.lax.stop_gradient(jnp.abs(q).mean())
        policy_loss = -lmbda * q.mean()
        return lmbda, policy_loss

      lmbda, guide_loss = jax.lax.cond(
        jax.random.uniform(rng) > 0.5, partial(fn, 'qf1'), partial(fn, 'qf2')
      )

      policy_loss = diff_loss + self.config.guide_coef * guide_loss
      return (policy_loss,), locals()

    # Calculat q losses and grads
    params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_qf), grads_qf = value_and_multi_grad(
      value_loss_fn, 2, has_aux=True
    )(params, tgt_params, rng)

    # Calculat policy losses and grads
    params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_policy), grads_policy = value_and_multi_grad(
      policy_loss_fn, 1, has_aux=True
    )(params, tgt_params, rng)

    # Update qf train states
    train_states['qf1'] = train_states['qf1'].apply_gradients(
      grads=grads_qf[0]['qf1']
    )
    train_states['qf2'] = train_states['qf2'].apply_gradients(
      grads=grads_qf[1]['qf2']
    )

    # Update policy train states
    train_states['policy'] = train_states['policy'].apply_gradients(
      grads=grads_policy[0]['policy']
    )

    # Update target parameters
    if policy_tgt_update:
      tgt_params['policy'] = update_target_network(
        train_states['policy'].params, tgt_params['policy'], self.config.tau
      )
    tgt_params['qf1'] = update_target_network(
      train_states['qf1'].params, tgt_params['qf1'], self.config.tau
    )
    tgt_params['qf2'] = update_target_network(
      train_states['qf2'].params, tgt_params['qf2'], self.config.tau
    )

    metrics = dict(
      qf_loss=aux_qf['qf_loss'],
      qf1_loss=aux_qf['qf1_loss'],
      qf2_loss=aux_qf['qf2_loss'],
      cur_q1=aux_qf['cur_q1'].mean(),
      cur_q2=aux_qf['cur_q2'].mean(),
      tgt_q1=aux_qf['tgt_q1'].mean(),
      tgt_q2=aux_qf['tgt_q2'].mean(),
      tgt_q=aux_qf['tgt_q'].mean(),
      policy_loss=aux_policy['policy_loss'],
      guide_loss=aux_policy['guide_loss'],
      diff_loss=aux_policy['diff_loss'],
      lmbda=aux_policy['lmbda'].mean(),
      qf1_grad_norm=optax.global_norm(grads_qf[0]['qf1']),
      qf2_grad_norm=optax.global_norm(grads_qf[1]['qf2']),
      policy_grad_norm=optax.global_norm(grads_policy[0]['policy']),
      qf1_weight_norm=optax.global_norm(train_states['qf1'].params),
      qf2_weight_norm=optax.global_norm(train_states['qf2'].params),
      policy_weight_norm=optax.global_norm(train_states['policy'].params),
    )

    return train_states, tgt_params, metrics

  def _train_step_crr(
    self, train_states, tgt_params, rng, batch, policy_tgt_update=False
  ):
    value_loss_fn = self.get_value_loss(batch)
    diff_loss_fn = self.get_diff_loss(batch)

    def policy_loss_fn(params, tgt_params, rng):
      observations = batch['observations']
      actions = batch['actions']

      rng, split_rng = jax.random.split(rng)
      # Calculate the guide loss
      diff_loss, terms, ts, pred_astart = diff_loss_fn(params, split_rng)

      # Construct the policy distribution
      action_dist = self.policy_dist.apply(params['policy_dist'], pred_astart)
      if self.config.crr_fixed_std:
        action_dist = distrax.MultivariateNormalDiag(
          pred_astart, jnp.ones_like(pred_astart)
        )

      # Build action distribution
      replicated_obs = jnp.broadcast_to(
        observations, (self.config.sample_actions,) + observations.shape
      )
      rng, split_rng = jax.random.split(rng)
      if self.config.use_pred_astart:
        vf_actions = action_dist.sample(
          seed=split_rng, sample_shape=self.config.sample_actions
        )
      else:
        vf_actions = self.policy.apply(
          params['policy'], split_rng, replicated_obs
        )

      # Compute the current Q estimates
      cur_q1 = self.qf.apply(params['qf1'], observations, actions)
      cur_q2 = self.qf.apply(params['qf2'], observations, actions)

      # Compute values
      v1 = self.qf.apply(params['qf1'], replicated_obs, vf_actions)
      v2 = self.qf.apply(params['qf2'], replicated_obs, vf_actions)
      v = jnp.minimum(v1, v2)
      q_pred = jnp.minimum(cur_q1, cur_q2)
      avg_fn = getattr(jnp, self.config.crr_avg_fn)
      adv = q_pred - avg_fn(v, axis=0)
      if self.config.crr_fn == 'exp':
        lmbda = jnp.minimum(
          self.config.crr_ratio_upper_bound,
          jnp.exp(adv / self.config.crr_beta)
        )
        if self.config.crr_adv_norm:
          lmbda = jax.nn.softmax(adv / self.config.crr_beta)
      else:
        lmbda = jnp.heaviside(adv, 0)
      lmbda = jax.lax.stop_gradient(lmbda)
      if self.config.crr_weight_mode == 'elbo':
        log_prob = -terms['ts_weights'] * terms['mse']
      elif self.config.crr_weight_mode == 'mle':
        log_prob = action_dist.log_prob(actions)
      else:
        rng, split_rng = jax.random.split(rng)
        if not self.config.crr_multi_sample_mse:
          sampled_actions = action_dist.sample(seed=split_rng)
          log_prob = -((sampled_actions - actions)**2).mean(axis=-1)
        else:
          sampled_actions = action_dist.sample(
            seed=split_rng, sample_shape=self.config.sample_actions
          )
          log_prob = -(
            (sampled_actions - jnp.expand_dims(actions, axis=0))**2
          ).mean(axis=(0, -1))
      guide_loss = -jnp.mean(log_prob * lmbda)

      policy_loss = self.config.diff_coef * diff_loss + \
        self.config.guide_coef * guide_loss
      losses = {'policy': policy_loss, 'policy_dist': policy_loss}
      return tuple(losses[key] for key in losses.keys()), locals()

    # Calculat policy losses and grads
    params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_qf), grads_qf = value_and_multi_grad(
      value_loss_fn, 2, has_aux=True
    )(params, tgt_params, rng)

    (_, aux_policy), grads_policy = value_and_multi_grad(
      policy_loss_fn, 2, has_aux=True
    )(params, tgt_params, rng)

    # Update qf train states
    train_states['qf1'] = train_states['qf1'].apply_gradients(
      grads=grads_qf[0]['qf1']
    )
    train_states['qf2'] = train_states['qf2'].apply_gradients(
      grads=grads_qf[1]['qf2']
    )

    train_states['policy'] = train_states['policy'].apply_gradients(
      grads=grads_policy[0]['policy']
    )
    train_states['policy_dist'] = train_states['policy_dist'].apply_gradients(
      grads=grads_policy[1]['policy_dist']
    )

    # Update target parameters
    if policy_tgt_update:
      tgt_params['policy'] = update_target_network(
        train_states['policy'].params, tgt_params['policy'], self.config.tau
      )
    tgt_params['qf1'] = update_target_network(
      train_states['qf1'].params, tgt_params['qf1'], self.config.tau
    )
    tgt_params['qf2'] = update_target_network(
      train_states['qf2'].params, tgt_params['qf2'], self.config.tau
    )

    metrics = dict(
      qf_loss=aux_qf['qf_loss'],
      qf1_loss=aux_qf['qf1_loss'],
      qf2_loss=aux_qf['qf2_loss'],
      cur_q1=aux_qf['cur_q1'].mean(),
      cur_q2=aux_qf['cur_q2'].mean(),
      tgt_q1=aux_qf['tgt_q1'].mean(),
      tgt_q2=aux_qf['tgt_q2'].mean(),
      tgt_q=aux_qf['tgt_q'].mean(),
      policy_loss=aux_policy['policy_loss'],
      guide_loss=aux_policy['guide_loss'],
      diff_loss=aux_policy['diff_loss'],
      lmbda=aux_policy['lmbda'].mean(),
      qf1_grad_norm=optax.global_norm(grads_qf[0]['qf1']),
      qf2_grad_norm=optax.global_norm(grads_qf[1]['qf2']),
      policy_grad_norm=optax.global_norm(grads_policy[0]['policy']),
      qf1_weight_norm=optax.global_norm(train_states['qf1'].params),
      qf2_weight_norm=optax.global_norm(train_states['qf2'].params),
      policy_weight_norm=optax.global_norm(train_states['policy'].params),
    )

    if self.config.loss_type == 'CRR':
      metrics['adv'] = aux_policy['adv'].mean()
      metrics['log_prob'] = aux_policy['log_prob'].mean()

    return train_states, tgt_params, metrics

  def train(self, batch):
    self._total_steps += 1
    policy_tgt_update = (
      self._total_steps > 1000 and
      self._total_steps % self.config.policy_tgt_freq == 0
    )
    self._train_states, self._tgt_params, metrics = self._train_step(
      self._train_states, self._tgt_params, next_rng(), batch,
      policy_tgt_update
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
