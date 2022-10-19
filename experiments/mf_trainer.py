import absl.flags
import gym
import numpy as np
import tqdm
from flax import linen as nn

import algos
from algos.model import (
  FullyConnectedQFunction,
  SamplerPolicy,
  TanhGaussianPolicy,
)
from data import Dataset, RandSampler
from experiments.args import FLAGS_DEF
from experiments.constants import ALGO, ALGO_MAP, ENV, ENV_MAP, ENVNAME_MAP
from utilities.jax_utils import batch_to_jax
from utilities.replay_buffer import get_d4rl_dataset
from utilities.sampler import TrajSampler
from utilities.utils import (
  Timer,
  WandBLogger,
  get_user_flags,
  norm_obs,
  prefix_metrics,
  set_random_seed,
)
from viskit.logging import logger, setup_logger


class MFTrainer:

  def __init__(self):
    self._cfgs = absl.flags.FLAGS
    self.algo = getattr(algos, self._cfgs.algo)

    self.variant = get_user_flags(self._cfgs, FLAGS_DEF)
    for k, v in self._cfgs.algo_cfg.items():
      self. variant[f"algo.{k}"] = v

    # get high level env
    env_name_full = self._cfgs.env
    for scenario_name in ENV_MAP:
      if scenario_name in env_name_full:
        self.env = ENV_MAP[scenario_name]
        break
    else:
      raise NotImplementedError

    set_random_seed(self._cfgs.seed)
    self.obs_mean = 0
    self.obs_std = 1
    self.obs_clip = np.inf

    self.eval_sampler = TrajSampler(gym.make(env_name_full), self._cfgs.max_traj_length)
    self.observation_dim = self.eval_sampler.env.observation_space.shape[0]
    self.action_dim = self.eval_sampler.env.action_space.shape[0]

    if self._cfgs.algo_cfg.target_entropy >= 0.0:
      action_space = self.eval_sampler.env.action_space
      self._cfgs.algo_cfg.target_entropy = -np.prod(action_space.shape).item()

    self.wandb_logger: WandBLogger = None
    self.dataset: Dataset = None
    self.policy: nn.Module = None
    self.qf: nn.Module = None
    self.agent: object = None
    self.sampler_policy: SamplerPolicy = None

  
  def train(self):

    self.__setup()

    viskit_metrics = {}
    for epoch in range(self._cfgs.n_epochs):
      metrics = {"epoch": epoch}

      with Timer() as train_timer:
        for _ in tqdm.tqdm(range(self._cfgs.n_train_step_per_epoch)):
          batch = batch_to_jax(self.dataset.sample())
          metrics.update(
            prefix_metrics(
              self.agent.train(batch, bc=epoch < self._cfgs.bc_epochs), "agent"
            )
          )

      with Timer() as eval_timer:
        if epoch == 0 or (epoch + 1) % self._cfgs.eval_period == 0:
          trajs = self.eval_sampler.sample(
            self.sampler_policy.update_params(self.agent.train_params["policy"]),
            self._cfgs.eval_n_trajs,
            deterministic=True,
            obs_statistics=(self.obs_mean, self.obs_std, self.obs_clip),
          )

          metrics["average_return"] = np.mean(
            [np.sum(t["rewards"]) for t in trajs]
          )
          metrics["average_traj_length"] = np.mean(
            [len(t["rewards"]) for t in trajs]
          )
          metrics["average_normalizd_return"] = np.mean(
            [
              self.eval_sampler.env.get_normalized_score(np.sum(t["rewards"]))
              for t in trajs
            ]
          )
          metrics["done"] = np.mean([np.sum(t["dones"]) for t in trajs])

          if self._cfgs.save_model:
            save_data = {"agent": self.agent, "variant": self.variant, "epoch": epoch}
            self.wandb_logger.save_pickle(save_data, f"model_{epoch}.pkl")

      metrics["train_time"] = train_timer()
      metrics["eval_time"] = eval_timer()
      metrics["epoch_time"] = train_timer() + eval_timer()
      self.wandb_logger.log(metrics)
      viskit_metrics.update(metrics)
      logger.record_dict(viskit_metrics)
      logger.dump_tabular(with_prefix=False, with_timestamp=False)

    # save model
    if self._cfgs.save_model:
      save_data = {"agent": self.agent, "variant": self.variant, "epoch": epoch}
      self.wandb_logger.save_pickle(save_data, "model_final.pkl")

  
  def __setup(self):

    # setup logger
    self.wandb_logger = self.__setup_logger()

    # setup dataset
    self.dataset = self.__setup_dataset()

    # setup policy
    self.policy = self.__setup_policy()

    # setup Q-function
    self.qf = self.__setup_qf()

    # setup agent
    self.agent = self.algo(self._cfgs.algo_cfg, self.policy, self.qf)

    # setup sampler policy
    self.sampler_policy = SamplerPolicy(self.agent.policy, self.agent.train_params["policy"])


  def __setup_logger(self):

    env_name_high = ENVNAME_MAP[self.env] 
    env_name_full = self._cfgs.env

    logging_configs = self._cfgs.logging
    logging_configs["project"] = f"MISA-{env_name_high}"
    wandb_logger = WandBLogger(
        config=logging_configs, variant=self.variant, env_name=env_name_full
    )
    setup_logger(
      variant=self.variant,
      exp_id=wandb_logger.experiment_id,
      seed=self._cfgs.seed,
      base_log_dir=self._cfgs.logging.output_dir,
      include_exp_prefix_sub_dir=False,
    )

    return wandb_logger

  
  def __setup_dataset(self):
    if self._cfgs.dataset == 'd4rl':
      dataset = get_d4rl_dataset(
        self.eval_sampler.env,
        self._cfgs.algo_cfg.nstep,
        self._cfgs.algo_cfg.discount,
      )

      dataset["rewards"] = dataset["rewards"] * self._cfgs.reward_scale + self._cfgs.reward_bias
      dataset["actions"] = np.clip(
        dataset["actions"], -self._cfgs.clip_action, self._cfgs.clip_action
      )

      if self.env == ENV.Kitchen or self.env == ENV.Adroit or self.env == ENV.Antmaze:
        if self._cfgs.obs_norm:
          self.obs_mean = dataset["observations"].mean()
          self.obs_std = dataset["observations"].std()
          self.obs_clip = 10
        norm_obs(dataset, self.obs_mean, self.obs_std, self.obs_clip)

        if self.env == ENV.Antmaze:
          dataset["rewards"] = (dataset["rewards"] - 0.5) * 4
        else:
          min_r, max_r = np.min(dataset["rewards"]), np.max(dataset["rewards"])
          dataset["rewards"] = (dataset["rewards"] - min_r) / (max_r - min_r)
          dataset["rewards"] = (dataset["rewards"] - 0.5) * 2

      dataset = Dataset(dataset)
      sampler = RandSampler(dataset.size(), self._cfgs.batch_size)
      dataset.set_sampler(sampler)

      return dataset


  def __setup_policy(self):
    if ALGO_MAP[self._cfgs.algo] == ALGO.MISA:
      policy = TanhGaussianPolicy(
        self.observation_dim,
        self.action_dim,
        self._cfgs.policy_arch,
        self._cfgs.orthogonal_init,
        self._cfgs.policy_log_std_multiplier,
        self._cfgs.policy_log_std_offset,
        use_layer_norm=self._cfgs.use_layer_norm,
      )
    else:
      raise NotImplementedError

    return policy

  
  def __setup_qf(self):
    if ALGO_MAP[self._cfgs.algo] == ALGO.MISA:
      qf = FullyConnectedQFunction(
        self.observation_dim,
        self.action_dim,
        self._cfgs.qf_arch,
        self._cfgs.orthogonal_init,
        self._cfgs.use_layer_norm,
        self._cfgs.activation,
      )
    else:
      raise NotImplementedError

    return qf
