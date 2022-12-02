from functools import partial
import absl
import absl.flags
import jax
import jax.numpy as jnp
import tqdm
import numpy as np

from experiments.constants import ENV_MAP
from utilities.utils import (
  Timer,
  WandBLogger,
  get_user_flags,
  define_flags_with_default,
  prefix_metrics,
)
from utilities.jax_utils import next_rng, batch_to_jax

from experiments.mf_trainer import MFTrainer
from diffusion.dql import DiffusionQL
from diffusion.nets import DiffusionPolicy, Critic
from diffusion.diffusion import (
  GaussianDiffusion, ModelMeanType, ModelVarType, LossType
)
from viskit.logging import logger
from diffusion.hps import hyperparameters

FLAGS_DEF = define_flags_with_default(
  algo="DiffusionQL",
  type="model-free",
  env="walker2d-medium-expert-v2",
  dataset='d4rl',
  rl_unplugged_task_class='control_suite',
  max_traj_length=1000,
  save_model=False,
  seed=42,
  batch_size=256,
  reward_scale=1,
  reward_bias=0,
  clip_action=0.999,
  encoder_arch="64-64",
  policy_arch="256-256-256",
  qf_arch="256-256-256",
  orthogonal_init=False,
  policy_log_std_multiplier=1.0,
  policy_log_std_offset=-1.0,
  algo_cfg=DiffusionQL.get_default_config(),
  n_epochs=1200,
  n_train_step_per_epoch=1000,
  eval_period=10,
  eval_n_trajs=10,
  logging=WandBLogger.get_default_config(),
  use_layer_norm=False,
  activation="elu",
  obs_norm=False,
)


class SamplerPolicy(object):

  def __init__(self, policy, qf=None, mean=0, std=1, ensemble=False):
    self.policy = policy
    self.qf = qf
    self.mean = mean
    self.std = std
    self.num_samples = 50
    self.ensemble = ensemble

  def update_params(self, params):
    self.params = params
    return self

  @partial(jax.jit, static_argnames=("self", "deterministic"))
  def act(self, params, rng, observations, deterministic):
    return self.policy.apply(
      params["policy"], rng, observations, deterministic, repeat=None
    )

  @partial(jax.jit, static_argnames=("self", "deterministic", "num_samples"))
  def ensemble_act(
    self, params, rng, observations, deterministic, num_samples
  ):
    rng, key = jax.random.split(rng)
    actions = self.policy.apply(
      params["policy"], key, observations, deterministic, repeat=num_samples
    )
    q1 = self.qf.apply(params['qf1'], observations, actions)
    q2 = self.qf.apply(params['qf2'], observations, actions)
    q = jnp.minimum(q1, q2)

    idx = jax.random.categorical(rng, q)
    return jnp.take(actions, idx, axis=-2)

  def __call__(self, observations, deterministic=False):
    observations = (observations - self.mean) / self.std
    if self.ensemble:
      actions = self.ensemble_act(
        self.params, next_rng(), observations, deterministic, self.num_samples
      )
    else:
      actions = self.act(
        self.params, next_rng(), observations, deterministic=deterministic
      )
    if isinstance(actions, tuple):
      actions = actions[0]
    assert jnp.all(jnp.isfinite(actions))
    return jax.device_get(actions)


class DiffusionTrainer(MFTrainer):

  def __init__(self):
    self._cfgs = absl.flags.FLAGS
    self._algo = DiffusionQL
    self._algo_type = 'DiffusionQL'
    self._cfgs.algo_cfg.max_grad_norm = hyperparameters[self._cfgs.env]['gn']
    self._cfgs.algo_cfg.lr_decay_steps = \
      self._cfgs.n_epochs * self._cfgs.n_train_step_per_epoch

    self._variant = get_user_flags(self._cfgs, FLAGS_DEF)
    for k, v in self._cfgs.algo_cfg.items():
      self._variant[f"algo.{k}"] = v

    # get high level env
    env_name_full = self._cfgs.env
    for scenario_name in ENV_MAP:
      if scenario_name in env_name_full:
        self._env = ENV_MAP[scenario_name]
        break
    else:
      raise NotImplementedError

  def train(self):
    self._setup()

    viskit_metrics = {}
    for epoch in range(self._cfgs.n_epochs):
      metrics = {"epoch": epoch}

      with Timer() as train_timer:
        for _ in tqdm.tqdm(range(self._cfgs.n_train_step_per_epoch)):
          batch = batch_to_jax(self._dataset.sample())
          metrics.update(prefix_metrics(self._agent.train(batch), "agent"))

      with Timer() as eval_timer:
        if epoch == 0 or (epoch + 1) % self._cfgs.eval_period == 0:
          self._sampler_policy.ensemble = True
          trajs_normal = self._eval_sampler.sample(
            self._sampler_policy.update_params(self._agent.train_params),
            self._cfgs.eval_n_trajs,
            deterministic=True,
            obs_statistics=(self._obs_mean, self._obs_std, self._obs_clip),
          )
          self._sampler_policy.ensemble = True
          trajs_ensemble = self._eval_sampler.sample(
            self._sampler_policy.update_params(self._agent.train_params),
            self._cfgs.eval_n_trajs,
            deterministic=True,
            obs_statistics=(self._obs_mean, self._obs_std, self._obs_clip),
          )

          for post, trajs in zip(['', '_ens'], [trajs_normal, trajs_ensemble]):
            metrics["average_return" +
                    post] = np.mean([np.sum(t["rewards"]) for t in trajs])
            metrics["average_traj_length" +
                    post] = np.mean([len(t["rewards"]) for t in trajs])
            metrics["average_normalizd_return" + post] = np.mean(
              [
                self._eval_sampler.env.get_normalized_score(
                  np.sum(t["rewards"])
                ) for t in trajs
              ]
            )
            metrics["done" +
                    post] = np.mean([np.sum(t["dones"]) for t in trajs])

          if self._cfgs.save_model:
            save_data = {
              "agent": self._agent,
              "variant": self._variant,
              "epoch": epoch
            }
            self._wandb_logger.save_pickle(save_data, f"model_{epoch}.pkl")

      metrics["train_time"] = train_timer()
      metrics["eval_time"] = eval_timer()
      metrics["epoch_time"] = train_timer() + eval_timer()
      self._wandb_logger.log(metrics)
      viskit_metrics.update(metrics)
      logger.record_dict(viskit_metrics)
      logger.dump_tabular(with_prefix=False, with_timestamp=False)

    # save model
    if self._cfgs.save_model:
      save_data = {
        "agent": self._agent,
        "variant": self._variant,
        "epoch": epoch
      }
      self._wandb_logger.save_pickle(save_data, "model_final.pkl")

  def _setup(self):
    super()._setup()
    # setup sampler policy
    self._sampler_policy = SamplerPolicy(self._agent.policy, self._agent.qf)

  def _setup_qf(self):
    qf = Critic(
      self._observation_dim,
      self._action_dim,
      to_arch(self._cfgs.qf_arch),
      use_layer_norm=self._cfgs.use_layer_norm
    )
    return qf

  def _setup_policy(self):
    gd = GaussianDiffusion(
      num_timesteps=self._cfgs.algo_cfg.num_timesteps,
      schedule_name=self._cfgs.algo_cfg.schedule_name,
      model_mean_type=ModelMeanType.EPSILON,
      model_var_type=ModelVarType.FIXED_SMALL,
      loss_type=LossType.MSE,
      min_value=-self._max_action,
      max_value=self._max_action,
    )
    policy = DiffusionPolicy(
      diffusion=gd,
      observation_dim=self._observation_dim,
      action_dim=self._action_dim,
      arch=to_arch(self._cfgs.policy_arch),
      time_embed_size=self._cfgs.algo_cfg.time_embed_size,
      use_layer_norm=self._cfgs.use_layer_norm,
    )

    return policy


def to_arch(string):
  return tuple(int(x) for x in string.split('-'))


if __name__ == '__main__':

  def main(argv):
    trainer = DiffusionTrainer()
    trainer.train()

  absl.app.run(main)
