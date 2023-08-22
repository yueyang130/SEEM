from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import optax
import math

from flax.training.train_state import TrainState
from ml_collections import ConfigDict
from core.core_api import Algo
from utilities.jax_utils import mse_loss, next_rng, value_and_multi_grad

from diffusion.diffusion import GaussianDiffusion
import distrax

def update_target_network(main_params, target_params, tau):
    return jax.tree_map(
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
        cfg.guide_warmup = False
        cfg.diff_annealing = False

        # learning related
        cfg.lr = 3e-4
        cfg.diff_coef = 1.0
        cfg.guide_coef = 1.0
        cfg.lr_decay = False
        cfg.train_steps = 1000000
        cfg.lr_decay_steps = 1000000
        cfg.max_grad_norm = 0.
        cfg.weight_decay = 0.

        cfg.loss_type = 'Rainbow'
        cfg.target_clip = False
        cfg.trust_region_target = False
        cfg.MAX_Q = 0.0
        cfg.use_expectile = False  # False: CRR; True: IQL
        cfg.expectile_q = False  # use td of expectile v to estimate q

        cfg.adv_norm = False
        # CRR-related hps
        cfg.sample_actions = 20
        cfg.crr_weight_mode = 'mle'
        cfg.fixed_std = True
        cfg.crr_multi_sample_mse = False
        cfg.crr_avg_fn = 'mean'
        cfg.crr_fn = 'exp'

        # IQL-related hps
        cfg.expectile = 0.7

        # CRR and IQL shared hps
        cfg.crr_ratio_upper_bound = 20
        cfg.crr_beta = 1.0
        cfg.awr_temperature = 3.0

        # reset
        cfg.reset_q = False
        cfg.reset_mode = 'all' # all / last / SP
        cfg.reset_actor = False
        cfg.reset_interval = 1000000
        cfg.max_tgt_q = False # update actor by maximizing target q
        
        # for dpm-solver
        cfg.dpm_steps = 15
        cfg.dpm_t_end = 0.001

        # useless
        cfg.target_entropy = -1
        if updates is not None:
            cfg.update(ConfigDict(updates).copy_and_resolve_references())
        return cfg

    def __init__(self, cfg, policy, qf, vf, policy_dist):
        self.config = self.get_default_config(cfg)
        self.policy = policy
        self.qf = qf
        self.vf = vf
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
            params=policy_dist_params, tx=get_optimizer(weight_decay=0.0), apply_fn=None
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

        vf_params = self.vf.init(
            next_rng(),
            jnp.zeros((10, self.observation_dim))
        )

        self._train_states['qf1'] = TrainState.create(
            params=qf1_params, tx=get_optimizer(), apply_fn=None
        )
        self._train_states['qf2'] = TrainState.create(
            params=qf2_params, tx=get_optimizer(), apply_fn=None
        )
        self._train_states['vf'] = TrainState.create(
            params=vf_params, tx=get_optimizer(), apply_fn=None,
        )
        self._tgt_params = deepcopy(
            {
                'policy': policy_params,
                'qf1': qf1_params,
                'qf2': qf2_params,
                'vf': vf_params,
            }
        )
        self._model_keys = ('policy', 'qf1', 'qf2', 'vf', 'policy_dist')



    def get_critic_loss(self, batch):
        
        def critic_loss_fn(params, tgt_params, rng):
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

            if self.config.target_clip:
                tgt_q = jnp.minimum(tgt_q, self.config.MAX_Q)
            if self.config.trust_region_target:  
                w = jnp.where((tgt_q<=self.config.MAX_Q) | dones.astype(bool), 1, 0)

            tgt_q = rewards + (1 - dones) * self.config.discount * tgt_q
            tgt_q = jax.lax.stop_gradient(tgt_q)

            # Compute the current Q estimates
            cur_q1 = self.qf.apply(params['qf1'], observations, actions)
            cur_q2 = self.qf.apply(params['qf2'], observations, actions)

            # qf loss
            if self.config.trust_region_target:  
                qf1_loss = jnp.mean(jnp.square(w*(cur_q1 - tgt_q)))
                qf2_loss = jnp.mean(jnp.square(w*(cur_q2 - tgt_q)))
            else:
                qf1_loss = mse_loss(cur_q1, tgt_q)
                qf2_loss = mse_loss(cur_q2, tgt_q)

            qf_loss = qf1_loss + qf2_loss
            return (qf1_loss, qf2_loss), locals()
  
        return critic_loss_fn

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



    # @partial(jax.jit, static_argnames=('self', 'policy_tgt_update', 'guide_warmup_coef', 'qf_update'))
    @partial(jax.jit, static_argnames=('self', 'policy_tgt_update', 'qf_update', 'q_tgt_update'))
    def _train_step(
        self, train_states, tgt_params, rng, batch, qf_batch, guide_warmup_coef, diff_coff, qf_update=False, policy_tgt_update=False, q_tgt_update=True,
    ):
        if self.config.loss_type not in ['Rainbow']:
            raise NotImplementedError

        return getattr(self, f"_train_step_{self.config.loss_type.lower()}"
                                    )(train_states, tgt_params, rng, batch, qf_batch, guide_warmup_coef, diff_coff, qf_update, policy_tgt_update, q_tgt_update)

    def _train_step_rainbow(
        self, train_states, tgt_params, rng, batch, qf_batch, guide_warmup_coef, diff_coff, qf_update=False, policy_tgt_update=False, q_tgt_update=True
    ):
        critic_loss_fn = self.get_critic_loss(qf_batch)
        diff_loss_fn = self.get_diff_loss(batch)

        def direct_guide_loss_fn(params, tgt_params, rng, pred_astart):
            observations = batch['observations']

            # Calculate guide loss
            def fn(key):
                if self.config.max_tgt_q:
                    q = self.qf.apply(tgt_params[key], observations, pred_astart)
                else:
                    q = self.qf.apply(params[key], observations, pred_astart)
                lmbda = 1 / jax.lax.stop_gradient(jnp.abs(q).mean())
                policy_loss = -lmbda * q.mean()
                return lmbda, policy_loss

            lmbda, guide_loss = jax.lax.cond(
                jax.random.uniform(rng) > 0.5, partial(fn, 'qf1'), partial(fn, 'qf2')
            )
            return guide_loss

     
        def policy_loss_fn(params, tgt_params, rng):

            rng, split_rng = jax.random.split(rng)
            diff_loss, terms, ts, pred_astart = diff_loss_fn(params, split_rng)
            td3_loss = direct_guide_loss_fn(params, tgt_params, rng, pred_astart)

            policy_loss = diff_coff * diff_loss + \
                        self.config.alpha * td3_loss * guide_warmup_coef

            return (policy_loss,), locals()

        if qf_update:
            # Calculat q losses and grads
            params = {key: train_states[key].params for key in self.model_keys}
            (_, aux_qf), grads_qf = value_and_multi_grad(
                critic_loss_fn, 2, has_aux=True
            )(params, tgt_params, rng)

            # Update qf train states
            train_states['qf1'] = train_states['qf1'].apply_gradients(
                grads=grads_qf[0]['qf1']
            )
            train_states['qf2'] = train_states['qf2'].apply_gradients(
                grads=grads_qf[1]['qf2']
            )

        # Calculat policy losses and grads
        params = {key: train_states[key].params for key in self.model_keys}
        (_, aux_policy), grads_policy = value_and_multi_grad(
            policy_loss_fn, 1, has_aux=True
        )(params, tgt_params, rng)

        # Update policy train states
        train_states['policy'] = train_states['policy'].apply_gradients(
            grads=grads_policy[0]['policy']
        )

        # Update target parameters
        if policy_tgt_update:
            tgt_params['policy'] = update_target_network(
                train_states['policy'].params, tgt_params['policy'], self.config.tau
            )
            
        if q_tgt_update:
            tgt_params['qf1'] = update_target_network(
                train_states['qf1'].params, tgt_params['qf1'], self.config.tau
            )
            tgt_params['qf2'] = update_target_network(
                train_states['qf2'].params, tgt_params['qf2'], self.config.tau
            )

        metrics = dict(
            policy_loss=aux_policy['policy_loss'],
            direct_loss=aux_policy['td3_loss'],
            diff_loss=aux_policy['diff_loss'],
            guide_warmup_coef=guide_warmup_coef,
            diff_coef=diff_coff,
            # lmbda=aux_policy['lmbda'].mean(),
            policy_grad_norm=optax.global_norm(grads_policy[0]['policy']),
            policy_weight_norm=optax.global_norm(train_states['policy'].params),
        )
  
        observations = qf_batch['observations']
        actions = qf_batch['actions']
        rng, split_rng = jax.random.split(rng)
        random_actions = jax.random.uniform(
          split_rng,
          shape=qf_batch['actions'].shape,
          minval=-self.max_action,
          maxval=self.max_action,
        )
        random_q = self.qf.apply(params['qf1'], observations, random_actions).mean()
        data_q = self.qf.apply(params['qf1'], observations, actions).mean()
        policy_q = aux_qf['cur_q1'].mean()
  
   
        if qf_update:
            metrics.update(dict(
                qf_loss=aux_qf['qf_loss'],
                qf1_loss=aux_qf['qf1_loss'],
                qf2_loss=aux_qf['qf2_loss'],
                cur_q1=aux_qf['cur_q1'].mean(),
                cur_q2=aux_qf['cur_q2'].mean(),
                tgt_q=aux_qf['tgt_q'].mean(),
                tgt_q1=aux_qf['tgt_q1'].mean(),
                tgt_q2=aux_qf['tgt_q2'].mean(),
                random_q=random_q,
                data_q=data_q,
                policy_q=policy_q,
                qf1_grad_norm=optax.global_norm(grads_qf[0]['qf1']),
                qf2_grad_norm=optax.global_norm(grads_qf[1]['qf2']),
                qf1_weight_norm=optax.global_norm(train_states['qf1'].params),
                qf2_weight_norm=optax.global_norm(train_states['qf2'].params),
                qf1_layer3_weight_norm=optax.global_norm(train_states['qf1'].params['params']['Dense_3']['kernel']),
                qf1_layer3_bias_norm=optax.global_norm(train_states['qf1'].params['params']['Dense_3']['bias']),
                qf1_layer2_weight_norm=optax.global_norm(train_states['qf1'].params['params']['Dense_2']['kernel']),
                qf1_layer2_bias_norm=optax.global_norm(train_states['qf1'].params['params']['Dense_2']['bias']),
                qf1_layer1_weight_norm=optax.global_norm(train_states['qf1'].params['params']['Dense_1']['kernel']),
                qf1_layer1_bias_norm=optax.global_norm(train_states['qf1'].params['params']['Dense_1']['bias']),
                qf1_layer0_weight_norm=optax.global_norm(train_states['qf1'].params['params']['Dense_0']['kernel']),
                qf1_layer0_bias_norm=optax.global_norm(train_states['qf1'].params['params']['Dense_0']['bias']),
                qf1_layer3_w_grad_norm=optax.global_norm(grads_qf[0]['qf1']['params']['Dense_3']['kernel']),
                qf1_layer3_b_grad_norm=optax.global_norm(grads_qf[0]['qf1']['params']['Dense_3']['bias']),
                qf1_layer2_w_grad_norm=optax.global_norm(grads_qf[0]['qf1']['params']['Dense_2']['kernel']),
                qf1_layer2_b_grad_norm=optax.global_norm(grads_qf[0]['qf1']['params']['Dense_2']['bias']),
                qf1_layer1_w_grad_norm=optax.global_norm(grads_qf[0]['qf1']['params']['Dense_1']['kernel']),
                qf1_layer1_b_grad_norm=optax.global_norm(grads_qf[0]['qf1']['params']['Dense_1']['bias']),
                qf1_layer0_w_grad_norm=optax.global_norm(grads_qf[0]['qf1']['params']['Dense_0']['kernel']),
                qf1_layer0_b_grad_norm=optax.global_norm(grads_qf[0]['qf1']['params']['Dense_0']['bias']),
                tgt_qf1_weight_norm=optax.global_norm(tgt_params['qf1']),
                tgt_qf2_weight_norm=optax.global_norm(tgt_params['qf1']),
            ))

        return train_states, tgt_params, metrics


    def train(self, batch, qf_batch):
        policy_tgt_update = (
            self._total_steps > 1000 and
            self._total_steps % self.config.policy_tgt_freq == 0
        )
        qf_update = True
        # reset
        if self._total_steps > 0 and self._total_steps % self.config.reset_interval == 0:
            if self.config.reset_q:
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
                self._train_states['qf1'] = self._train_states['qf1'].replace(params=qf1_params)
                self._train_states['qf2'] = self._train_states['qf2'].replace(params=qf2_params)
                self._tgt_params['qf1'] = deepcopy(qf1_params)
                self._tgt_params['qf2'] = deepcopy(qf2_params)
            if self.config.reset_actor:
                policy_params = self.policy.init(
                    next_rng(),
                    next_rng(),
                    jnp.zeros((10, self.observation_dim)),
                )
                policy_dist_params = self.policy_dist.init(
                    next_rng(), jnp.zeros((10, self.action_dim))
                )
                self._train_states['policy'] = self._train_states['policy'].replace(params=policy_params)
                self._train_states['policy_dist'] = self._train_states['policy_dist'].replace(params=policy_dist_params)
            
        q_tgt_update = True
        # if self.config.reset_q:
        #     reset_num = int(self._total_steps // self.config.reset_interval)
        #     if reset_num > 0 and self._total_steps % self.config.reset_interval < 100000:
        #         q_tgt_update = False
        
        self._train_states, self._tgt_params, metrics = self._train_step(
            self._train_states, self._tgt_params, next_rng(), batch, qf_batch,
            self.guide_warmup_coef, self.diff_annealing_coef, qf_update, policy_tgt_update, q_tgt_update
        )
        self._total_steps += 1
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
    
    # return a constant in SEEM experiments
    @property
    def guide_warmup_coef(self):
            if self.config.guide_warmup:
                if self._total_steps < self.config.train_steps // 4:
                    return 0
                elif self._total_steps < self.config.train_steps // 2:
                    return  4 * self._total_steps / self.config.train_steps - 1
                else:
                    return 1.0
                # return 0.0
            else:
                return 1.0
            
    # return a constant in SEEM experiments
    @property
    def diff_annealing_coef(self):
        if self.config.diff_annealing:
            t = self._total_steps
            T = self.config.constraint_steps
            if t <= T:
                return self.config.diff_coef * ( 0.5 + 0.5 * math.cos(math.pi * t / T))
            else:
                return 0
        else:
            return self.config.diff_coef