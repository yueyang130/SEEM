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
    cfg.policy_freq = 2
    cfg.num_timesteps = 100
    cfg.schedule_name = 'linear'
    cfg.time_embed_size = 16
    cfg.alpha = 2.  # NOTE 0.25 in diffusion rl but 2.5 in td3
    cfg.use_pred_astart = False
    cfg.max_grad_norm = 0.

    # learning related
    cfg.lr = 3e-4
    cfg.guide_coef = 1.0
    cfg.lr_decay = False
    cfg.lr_decay_steps = 1000000

    cfg.loss_type = 'TD3'
    cfg.use_expectile = False
    cfg.exp_tau = 0.7
    cfg.sample_actions = 20
    cfg.crr_ratio_upper_bound = 20
    cfg.crr_beta = 1.0

    # useless
    cfg.target_entropy = -1
    if updates is not None:
      cfg.update(ConfigDict(updates).copy_and_resolve_references())
    return cfg

  def __init__(self, cfg, policy, qf):
    self.config = self.get_default_config(cfg)
    self.policy = policy
    self.qf = qf
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

    def get_lr():
      if self.config.lr_decay is True:
        return optax.cosine_decay_schedule(
          self.config.lr, decay_steps=self.config.lr_decay_steps
        )
      else:
        return self.config.lr

    def get_optimizer():
      if self.config.max_grad_norm > 0:
        opt = optax.chain(
          optax.clip_by_global_norm(self.config.max_grad_norm),
          optax.adam(get_lr()),
        )
      else:
        opt = optax.adam(get_lr())

      return opt

    self._train_states['policy'] = TrainState.create(
      params=policy_params, tx=get_optimizer(), apply_fn=None
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
    self._model_keys = ('policy', 'qf1', 'qf2')

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
      next_action = self.policy.apply(
        tgt_params['policy'], split_rng, next_observations
      )
      # TODO: check whether we need to clip `next_action`
      next_action = jnp.clip(next_action, -self.max_action, self.max_action)

      # Compute the target Q values (without gradient)
      tgt_q1 = self.qf.apply(tgt_params['qf1'], next_observations, next_action)
      tgt_q2 = self.qf.apply(tgt_params['qf2'], next_observations, next_action)
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
        diff1 = cur_q1 - tgt_q
        exp_w1 = jnp.where(
          diff1 > 0, self.config.exp_tau, 1 - self.config.exp_tau
        )
        qf1_loss = (exp_w1 * (diff1 ** 2)).mean()
        diff2 = cur_q2 - tgt_q
        exp_w2 = jnp.where(
          diff2 > 0, self.config.exp_tau, 1 - self.config.exp_tau
        )
        qf2_loss = (exp_w2 * (diff2 ** 2)).mean()

      qf_loss = qf1_loss + qf2_loss

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

      rng, split_rng = jax.random.split(rng)
      replicated_out = jnp.broadcast_to(
        terms["model_output"], (self.config.sample_actions,) + terms["model_output"].shape
      )
      replicated_obs = jnp.broadcast_to(
        observations, (self.config.sample_actions,) + observations.shape
      )
      replicated_actions = jnp.broadcast_to(
        actions, (self.config.sample_actions,) + actions.shape
      )
      if self.config.use_pred_astart:
        pred_astart = self.diffusion.p_mean_variance(
          terms["model_output"], actions, ts
        )["pred_xstart"]
        vf_actions = self.diffusion.p_mean_variance(
          replicated_out, replicated_actions, jnp.zeros(
            (self.config.sample_actions,) + ts.shape, dtype=jnp.int32
          )
        )["pred_xstart"]
      else:
        pred_astart = self.policy.apply(
          params['policy'], split_rng, observations
        )
        rng, split_rng= jax.random.split(rng)
        vf_actions = self.policy.apply(
          params['policy'], split_rng, replicated_obs
        )

      v1 = self.qf.apply(params['qf1'], replicated_obs, vf_actions)
      v2 = self.qf.apply(params['qf2'], replicated_obs, vf_actions)
      v = jnp.minimum(v1, v2)

      if self.config.loss_type == 'TD3':
        def fn(key):
          q = self.qf.apply(params[key], observations, pred_astart)
          lmbda = self.config.alpha / jax.lax.stop_gradient(jnp.abs(q).mean())
          policy_loss = -lmbda * q.mean()
          return lmbda, policy_loss

        lmbda, guide_loss = jax.lax.cond(
          jax.random.uniform(rng) > 0.5, partial(fn, 'qf1'), partial(fn, 'qf2')
        )
        policy_loss = diff_loss + self.config.guide_coef * guide_loss
      elif self.config.loss_type == 'CRR':
        q_pred = jnp.minimum(cur_q1, cur_q2)
        adv = q_pred - jnp.mean(v, axis=0)
        coef = jnp.minimum(
          self.config.crr_ratio_upper_bound,
          jnp.exp(adv / self.config.crr_beta)
        )
        coef = jax.lax.stop_gradient(coef)
        action_dist = distrax.MultivariateNormalDiag(
          pred_astart, jnp.ones_like(pred_astart)
        )
        log_prob = action_dist.log_prob(actions)
        policy_loss = -jnp.mean(log_prob * coef)
      else:
        raise NotImplementedError

      losses = {'policy': policy_loss, 'qf1': qf1_loss, 'qf2': qf2_loss}
      return tuple(losses[key] for key in self.model_keys), locals()

    # Calculat losses and grads
    params = {key: train_states[key].params for key in self.model_keys}
    (_, aux_values), grads = value_and_multi_grad(
      loss_fn, len(self.model_keys), has_aux=True
    )(params, tgt_params, rng)

    # Update qf train states
    train_states['qf1'] = train_states['qf1'].apply_gradients(
      grads=grads[1]['qf1']
    )
    train_states['qf2'] = train_states['qf2'].apply_gradients(
      grads=grads[2]['qf2']
    )

    # Update policy train states if necessary
    if policy_update:
      train_states['policy'] = train_states['policy'].apply_gradients(
        grads=grads[0]['policy']
      )

      # Update target parameters
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
      policy_loss=aux_values['policy_loss'],
      qf_loss=aux_values['qf_loss'],
      qf1_loss=aux_values['qf1_loss'],
      qf2_loss=aux_values['qf2_loss'],
      guide_loss=aux_values['guide_loss'],
      diff_loss=aux_values['diff_loss'],
      cur_q1=aux_values['cur_q1'].mean(),
      cur_q2=aux_values['cur_q2'].mean(),
      tgt_q1=aux_values['tgt_q1'].mean(),
      tgt_q2=aux_values['tgt_q2'].mean(),
      tgt_q=aux_values['tgt_q'].mean(),
      lmbda=aux_values['lmbda'],
    )

    return train_states, tgt_params, metrics

  def train(self, batch):
    self._total_steps += 1
    self._train_states, self._tgt_params, metrics = self._train_step(
      self._train_states, self._tgt_params, next_rng(), batch,
      self._total_steps % self.config.policy_freq == 0
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
