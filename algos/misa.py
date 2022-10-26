# Copyright 2022 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
MISA learner.
"""

from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState
from ml_collections import ConfigDict
from tensorflow_probability.substrates import jax as tfp

from algos.model import Scalar, update_target_network
from utilities.jax_utils import mse_loss, next_rng, value_and_multi_grad
from utilities.utils import prefix_metrics


class MISA(object):

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
    config.optimizer_type = "adam"
    config.soft_target_update_rate = 5e-3
    config.use_cql = True
    config.n_actions = 50
    config.importance_sample = True
    config.lagrange = False
    config.target_action_gap = 3.0
    config.temp = 1.0
    config.min_q_weight = 5.0
    config.max_target_backup = False
    config.clip_diff_min = -np.inf
    config.clip_diff_max = np.inf
    config.bc_mode = "mle"  # 'mle'
    config.misa = True
    config.bc_weight_misa = 0.5
    config.unbiased_grad = True
    config.unbiased_weight = 0.1
    config.mcmc_burnin_steps = 5
    config.mcmc_num_leapfrog_steps = 2
    config.mcmc_step_size = 1
    config.log_k_norm = True
    config.detach_pi = False
    config.add_positive = False

    if updates is not None:
      config.update(ConfigDict(updates).copy_and_resolve_references())
    return config

  def __init__(self, config, policy, qf):
    self.config = self.get_default_config(config)
    self.policy = policy
    self.qf = qf
    self.observation_dim = policy.input_size
    self.action_dim = policy.action_dim

    self._train_states = {}

    optimizer_class = {
      "adam": optax.adam,
      "sgd": optax.sgd,
    }[self.config.optimizer_type]

    policy_params = self.policy.init(
      next_rng(), next_rng(), jnp.zeros((10, self.observation_dim))
    )
    self._train_states["policy"] = TrainState.create(
      params=policy_params,
      tx=optimizer_class(self.config.policy_lr),
      apply_fn=None,
    )

    qf1_params = self.qf.init(
      next_rng(),
      jnp.zeros((10, self.observation_dim)),
      jnp.zeros((10, self.action_dim)),
    )
    self._train_states["qf1"] = TrainState.create(
      params=qf1_params,
      tx=optimizer_class(self.config.qf_lr),
      apply_fn=None,
    )
    qf2_params = self.qf.init(
      next_rng(),
      jnp.zeros((10, self.observation_dim)),
      jnp.zeros((10, self.action_dim)),
    )
    self._train_states["qf2"] = TrainState.create(
      params=qf2_params,
      tx=optimizer_class(self.config.qf_lr),
      apply_fn=None,
    )
    self._target_qf_params = deepcopy({"qf1": qf1_params, "qf2": qf2_params})

    model_keys = ["policy", "qf1", "qf2"]

    if self.config.use_automatic_entropy_tuning:
      self.log_alpha = Scalar(0.0)
      self._train_states["log_alpha"] = TrainState.create(
        params=self.log_alpha.init(next_rng()),
        tx=optimizer_class(self.config.policy_lr),
        apply_fn=None,
      )
      model_keys.append("log_alpha")

    if self.config.lagrange:
      self.log_alpha_prime = Scalar(1.0)
      self._train_states["log_alpha_prime"] = TrainState.create(
        params=self.log_alpha_prime.init(next_rng()),
        tx=optimizer_class(self.config.qf_lr),
        apply_fn=None,
      )
      model_keys.append("log_alpha_prime")

    self._model_keys = tuple(model_keys)
    self._total_steps = 0

  def train(self, batch, bc=False):
    self._total_steps += 1
    self._train_states, self._target_qf_params, metrics = self._train_step(
      self._train_states, self._target_qf_params, next_rng(), batch, bc
    )
    return metrics

  @partial(jax.jit, static_argnames=("self", "bc"))
  def _train_step(self, train_states, target_qf_params, rng, batch, bc=False):

    def loss_fn(train_params, rng):
      observations = batch["observations"]
      actions = batch["actions"]
      rewards = batch["rewards"]
      next_observations = batch["next_observations"]
      dones = batch["dones"]

      loss_collection = {}

      rng, split_rng = jax.random.split(rng)
      new_actions, log_pi = self.policy.apply(
        train_params["policy"], split_rng, observations
      )

      if self.config.use_automatic_entropy_tuning:
        if self.config.detach_pi:
          alpha_loss = (
            -self.log_alpha.apply(train_params["log_alpha"]) *
            (jax.lax.stop_gradient(log_pi) + self.config.target_entropy).mean()
          )
        else:
          alpha_loss = (
            -self.log_alpha.apply(train_params["log_alpha"]) *
            (log_pi + self.config.target_entropy).mean()
          )

        loss_collection["log_alpha"] = alpha_loss
        alpha = (
          jnp.exp(self.log_alpha.apply(train_params["log_alpha"])) *
          self.config.alpha_multiplier
        )
      else:
        alpha_loss = 0.0
        alpha = self.config.alpha_multiplier
      """ Policy loss """
      # get bc loss
      if self.config.bc_mode == "mle":
        log_probs = self.policy.apply(
          train_params["policy"],
          observations,
          actions,
          method=self.policy.log_prob,
        )
        bc_loss = (alpha * log_pi - log_probs).mean()
      elif self.config.bc_mode == "mse":
        bc_loss = mse_loss(actions, new_actions)
      else:
        raise RuntimeError("{} not implemented!".format(self.config.bc_mode))

      # get (offline)rl loss
      if bc:
        rl_loss = bc_loss
      elif self.config.misa:
        bc_weight = self.config.bc_weight_misa
        rl_loss = bc_weight * bc_loss
        q_new_actions = jnp.minimum(
          self.qf.apply(train_params["qf1"], observations, new_actions),
          self.qf.apply(train_params["qf2"], observations, new_actions),
        )
        rl_loss += (-q_new_actions).mean()
      else:
        q_new_actions = jnp.minimum(
          self.qf.apply(train_params["qf1"], observations, new_actions),
          self.qf.apply(train_params["qf2"], observations, new_actions),
        )
        rl_loss = (alpha * log_pi - q_new_actions).mean()

      # total loss for policy
      policy_loss = rl_loss
      loss_collection["policy"] = policy_loss
      loss_collection["bc_loss"] = bc_loss
      loss_collection["rl_loss"] = rl_loss
      """ Q function loss """
      q1_pred = self.qf.apply(train_params["qf1"], observations, actions)
      q2_pred = self.qf.apply(train_params["qf2"], observations, actions)

      rng, split_rng = jax.random.split(rng)
      if self.config.max_target_backup:
        new_next_actions, next_log_pi = self.policy.apply(
          train_params["policy"],
          split_rng,
          next_observations,
          repeat=self.config.n_actions,
        )
        target_q_values = jnp.minimum(
          self.qf.apply(
            target_qf_params["qf1"], next_observations, new_next_actions
          ),
          self.qf.apply(
            target_qf_params["qf2"], next_observations, new_next_actions
          ),
        )
        max_target_indices = jnp.expand_dims(
          jnp.argmax(target_q_values, axis=-1), axis=-1
        )
        target_q_values = jnp.take_along_axis(
          target_q_values, max_target_indices, axis=-1
        ).squeeze(-1)
        next_log_pi = jnp.take_along_axis(
          next_log_pi, max_target_indices, axis=-1
        ).squeeze(-1)
      else:
        new_next_actions, next_log_pi = self.policy.apply(
          train_params["policy"], split_rng, next_observations
        )
        target_q_values = jnp.minimum(
          self.qf.apply(
            target_qf_params["qf1"], next_observations, new_next_actions
          ),
          self.qf.apply(
            target_qf_params["qf2"], next_observations, new_next_actions
          ),
        )

      if self.config.backup_entropy:
        target_q_values = target_q_values - alpha * next_log_pi

      discount = self.config.discount
      td_target = jax.lax.stop_gradient(
        rewards + (1.0 - dones) * discount * target_q_values
      )
      qf1_loss = mse_loss(q1_pred, td_target)
      qf2_loss = mse_loss(q2_pred, td_target)

      # CQL
      if self.config.use_cql:
        batch_size = actions.shape[0]
        rng, split_rng = jax.random.split(rng)
        random_actions = jax.random.uniform(
          split_rng,
          shape=(batch_size, self.config.n_actions, self.action_dim),
          minval=-1.0,
          maxval=1.0,
        )

        rng, split_rng = jax.random.split(rng)
        current_actions, current_log_pis = self.policy.apply(
          train_params["policy"],
          split_rng,
          observations,
          repeat=self.config.n_actions,
        )
        rng, split_rng = jax.random.split(rng)
        next_actions, next_log_pis = self.policy.apply(
          train_params["policy"],
          split_rng,
          next_observations,
          repeat=self.config.n_actions,
        )

        q1_rand = self.qf.apply(
          train_params["qf1"], observations, random_actions
        )
        q2_rand = self.qf.apply(
          train_params["qf2"], observations, random_actions
        )
        q1_current_actions = self.qf.apply(
          train_params["qf1"], observations, current_actions
        )
        q2_current_actions = self.qf.apply(
          train_params["qf2"], observations, current_actions
        )
        q1_next_actions = self.qf.apply(
          train_params["qf1"], observations, next_actions
        )
        q2_next_actions = self.qf.apply(
          train_params["qf2"], observations, next_actions
        )

        if self.config.misa:
          cat_q1 = q1_current_actions
          cat_q2 = q2_current_actions

          if self.config.add_positive:
            cat_q1 = jnp.concatenate(
              [cat_q1, jnp.expand_dims(q1_pred, axis=1)], axis=1
            )
            cat_q2 = jnp.concatenate(
              [cat_q2, jnp.expand_dims(q2_pred, axis=1)], axis=1
            )

        elif self.config.importance_sample:
          random_density = np.log(0.5**self.action_dim)
          if self.config.detach_pi:
            next_log_pi = jax.lax.stop_gradient(next_log_pis)
            cur_log_pi = jax.lax.stop_gradient(current_log_pis)
          else:
            next_log_pi = next_log_pis
            cur_log_pi = current_log_pis

          cat_q1 = jnp.concatenate(
            [
              q1_rand - random_density,
              q1_next_actions - next_log_pi,
              q1_current_actions - cur_log_pi,
            ],
            axis=1,
          )
          cat_q2 = jnp.concatenate(
            [
              q2_rand - random_density,
              q2_next_actions - next_log_pis,
              q2_current_actions - current_log_pis,
            ],
            axis=1,
          )
        else:
          cat_q1 = jnp.concatenate(
            [
              q1_rand,
              jnp.expand_dims(q1_pred, 1),
              q1_next_actions,
              q1_current_actions,
            ],
            axis=1,
          )
          cat_q2 = jnp.concatenate(
            [
              q2_rand,
              jnp.expand_dims(q2_pred, 1),
              q2_next_actions,
              q2_current_actions,
            ],
            axis=1,
          )

        if self.config.log_k_norm:
          cat_q1 = cat_q1 - jnp.log(cat_q1.shape[0])
          cat_q2 = cat_q2 - jnp.log(cat_q2.shape[0])

        std_q1 = jnp.std(cat_q1, axis=1)
        std_q2 = jnp.std(cat_q2, axis=1)

        qf1_ood = (
          jax.scipy.special.logsumexp(cat_q1 / self.config.temp, axis=1) *
          self.config.temp
        )
        qf2_ood = (
          jax.scipy.special.logsumexp(cat_q2 / self.config.temp, axis=1) *
          self.config.temp
        )
        """Subtract the log likelihood of data"""
        qf1_diff = jnp.clip(
          qf1_ood - q1_pred,
          self.config.clip_diff_min,
          self.config.clip_diff_max,
        ).mean()
        qf2_diff = jnp.clip(
          qf2_ood - q2_pred,
          self.config.clip_diff_min,
          self.config.clip_diff_max,
        ).mean()

        if self.config.lagrange:
          alpha_prime = jnp.clip(
            jnp.exp(
              self.log_alpha_prime.apply(train_params["log_alpha_prime"])
            ),
            a_min=0.0,
            a_max=1000000.0,
          )
          min_qf1_loss = (
            alpha_prime * self.config.min_q_weight *
            (qf1_diff - self.config.target_action_gap)
          )
          min_qf2_loss = (
            alpha_prime * self.config.min_q_weight *
            (qf2_diff - self.config.target_action_gap)
          )

          alpha_prime_loss = (-min_qf1_loss - min_qf2_loss) * 0.5

          loss_collection["log_alpha_prime"] = alpha_prime_loss

        else:
          min_qf1_loss = qf1_diff * self.config.min_q_weight
          min_qf2_loss = qf2_diff * self.config.min_q_weight
          alpha_prime_loss = 0.0
          alpha_prime = 0.0

        qf1_loss = qf1_loss + min_qf1_loss
        qf2_loss = qf2_loss + min_qf2_loss

        if self.config.misa and self.config.unbiased_grad:

          def log_prob(x):
            action_prob = self.policy.apply(
              train_params["policy"],
              observations,
              x,
              method=self.policy.log_prob,
            )
            q = jnp.minimum(
              self.qf.apply(train_params["qf1"], observations, x),
              self.qf.apply(train_params["qf2"], observations, x),
            )

            # we ignore the E_{\pi(a | s)}[exp(Q(s, a))] here
            # because this is a constant for a given s
            return action_prob + q

          num_results = self.config.n_actions
          num_burnin_steps = self.config.mcmc_burnin_steps
          adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
            tfp.mcmc.HamiltonianMonteCarlo(
              target_log_prob_fn=log_prob,
              num_leapfrog_steps=self.config.mcmc_num_leapfrog_steps,
              step_size=self.config.mcmc_step_size,
            ),
            num_adaptation_steps=int(num_burnin_steps * 0.8),
          )

          rng, split_rng = jax.random.split(rng)
          mcmc_action_samples, _ = tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=actions,
            kernel=adaptive_hmc,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
            seed=split_rng,
          )

          mcmc_action_samples = jax.lax.stop_gradient(
            jnp.transpose(mcmc_action_samples, (1, 0, 2))
          )

          correction_log_pis = self.policy.apply(
            train_params["policy"],
            observations,
            mcmc_action_samples,
            method=self.policy.log_prob,
          )
          loss_collection["policy"] += (
            correction_log_pis.mean() * self.config.unbiased_weight
          )

      loss_collection["qf1"] = qf1_loss
      loss_collection["qf2"] = qf2_loss
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
    new_target_qf_params["qf1"] = update_target_network(
      new_train_states["qf1"].params,
      target_qf_params["qf1"],
      self.config.soft_target_update_rate,
    )
    new_target_qf_params["qf2"] = update_target_network(
      new_train_states["qf2"].params,
      target_qf_params["qf2"],
      self.config.soft_target_update_rate,
    )

    metrics = dict(
      log_pi=aux_values["log_pi"].mean(),
      policy_loss=aux_values["policy_loss"],
      bc_loss=aux_values["bc_loss"],
      rl_loss=aux_values["rl_loss"],
      qf1_loss=aux_values["qf1_loss"],
      qf2_loss=aux_values["qf2_loss"],
      alpha_loss=aux_values["alpha_loss"],
      alpha=aux_values["alpha"],
      average_qf1=aux_values["q1_pred"].mean(),
      average_qf2=aux_values["q2_pred"].mean(),
      average_target_q=aux_values["target_q_values"].mean(),
    )

    if self.config.use_cql:
      metrics.update(
        prefix_metrics(
          dict(
            std_q1=aux_values["std_q1"].mean(),
            std_q2=aux_values["std_q2"].mean(),
            q1_rand=aux_values["q1_rand"].mean(),
            q2_rand=aux_values["q2_rand"].mean(),
            qf1_diff=aux_values["qf1_diff"].mean(),
            qf2_diff=aux_values["qf2_diff"].mean(),
            min_qf1_loss=aux_values["min_qf1_loss"].mean(),
            min_qf2_loss=aux_values["min_qf2_loss"].mean(),
            q1_current_actions=aux_values["q1_current_actions"].mean(),
            q2_current_actions=aux_values["q2_current_actions"].mean(),
            q1_next_actions=aux_values["q1_next_actions"].mean(),
            q2_next_actions=aux_values["q2_next_actions"].mean(),
            alpha_prime=aux_values["alpha_prime"],
            alpha_prime_loss=aux_values["alpha_prime_loss"],
          ),
          "cql",
        )
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
