from pathlib import Path

import absl.flags
import gym
import numpy as np
import tqdm
from flax import linen as nn

import algos
from algos.model import (
  DirectMappingPolicy,
  FullyConnectedQFunction,
  FullyConnectedVFunction,
  SamplerPolicy,
  TanhGaussianPolicy,
)
from core.core_api import Trainer
from data import Dataset, DM2Gym, RandSampler, RLUPDataset, PrefetchBalancedSampler
from experiments.constants import (
  ALGO,
  ALGO_MAP,
  DATASET,
  DATASET_ABBR_MAP,
  DATASET_MAP,
  ENV,
  ENV_MAP,
  ENVNAME_MAP,
)
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


class MFTrainer(Trainer):

  def __init__(self, flags_def):
    self._cfgs = absl.flags.FLAGS
    self._algo = getattr(algos, self._cfgs.algo)
    self._algo_type = ALGO_MAP[self._cfgs.algo]

    self._variant = get_user_flags(self._cfgs, flags_def)
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

    self._obs_mean: float = None
    self._obs_std: float = None
    self._obs_clip: float = None

    self._eval_sampler: TrajSampler = None
    self._observation_dim: int = None
    self._action_dim: int = None

    self._wandb_logger: WandBLogger = None
    self._dataset: Dataset = None
    self._policy: nn.Module = None
    self._qf: nn.Module = None
    self._vf: nn.Module = None
    self._agent: object = None
    self._sampler_policy: SamplerPolicy = None

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
          trajs = self._eval_sampler.sample(
            self._sampler_policy.update_params(
              self._agent.train_params["policy"]
            ),
            self._cfgs.eval_n_trajs,
            deterministic=True,
            obs_statistics=(self._obs_mean, self._obs_std, self._obs_clip),
          )

          metrics["average_return"] = np.mean(
            [np.sum(t["rewards"]) for t in trajs]
          )
          metrics["average_traj_length"] = np.mean(
            [len(t["rewards"]) for t in trajs]
          )
          metrics["average_normalizd_return"] = np.mean(
            [
              self._eval_sampler.env.get_normalized_score(
                np.sum(t["rewards"])
              ) for t in trajs
            ]
          )
          metrics["done"] = np.mean([np.sum(t["dones"]) for t in trajs])

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
    set_random_seed(self._cfgs.seed)

    # setup logger
    self._wandb_logger = self._setup_logger()

    # setup dataset and eval_sample
    self._dataset, self._eval_sampler = self._setup_dataset()

    # setup policy
    self._policy = self._setup_policy()

    # setup Q-function
    self._qf = self._setup_qf()

    # setup vf only for IQL
    if self._algo_type == ALGO.IQL:
      self._vf = self._setup_vf()

    # setup agent
    if self._algo_type == ALGO.IQL:
      self._agent = self._algo(
        self._cfgs.algo_cfg, self._policy, self._qf, self._vf
      )
    else:
      self._agent = self._algo(self._cfgs.algo_cfg, self._policy, self._qf)

    # setup sampler policy
    self._sampler_policy = SamplerPolicy(
      self._agent.policy, self._agent.train_params["policy"]
    )

  def _setup_logger(self):
    env_name_high = ENVNAME_MAP[self._env]
    env_name_full = self._cfgs.env
    dataset_name_abbr = DATASET_ABBR_MAP[self._cfgs.dataset]

    logging_configs = self._cfgs.logging
    logging_configs["project"
                   ] = f"{self._cfgs.algo}-{env_name_high}-{dataset_name_abbr}"
    wandb_logger = WandBLogger(
      config=logging_configs, variant=self._variant, env_name=env_name_full
    )
    setup_logger(
      variant=self._variant,
      exp_id=wandb_logger.experiment_id,
      seed=self._cfgs.seed,
      base_log_dir=self._cfgs.logging.output_dir,
      include_exp_prefix_sub_dir=False,
    )

    return wandb_logger

  def _setup_d4rl(self):
    eval_sampler = TrajSampler(
      gym.make(self._cfgs.env), self._cfgs.max_traj_length
    )

    norm_reward = self._cfgs.norm_reward
    if 'antmaze' in self._cfgs.env:
      norm_reward = False

    # OPER constant
    if self._env == ENV.Mujoco:
      ITER = 5
      STD = 2
      EPS = 0.1
      BASE_PROB = 0
    elif self._env == ENV.Antmaze:
      ITER = 3
      STD = 5
      EPS = 0.1
      BASE_PROB = 0.2
    elif self._env == ENV.Kitchen or self._env == ENV.Adroit:
      ITER = 4
      STD = 0.5
      EPS = 0.1
      BASE_PROB = 0

    dataset = get_d4rl_dataset(
      eval_sampler.env,
      self._cfgs.algo_cfg.nstep,
      self._cfgs.algo_cfg.discount,
      norm_reward=norm_reward,
    )
    dataset["rewards"] = dataset[
      "rewards"] * self._cfgs.reward_scale + self._cfgs.reward_bias
    dataset["actions"] = np.clip(
      dataset["actions"], -self._cfgs.clip_action, self._cfgs.clip_action
    )

    if self._env == ENV.Kitchen or self._env == ENV.Adroit or self._env == ENV.Antmaze:
      if self._cfgs.obs_norm:
        self._obs_mean = dataset["observations"].mean()
        self._obs_std = dataset["observations"].std()
        self._obs_clip = 10
      norm_obs(dataset, self._obs_mean, self._obs_std, self._obs_clip)

      if self._env == ENV.Antmaze:
        if self._cfgs.algo_cfg.loss_type in ['IQL', 'Rainbow']:
          # dataset["rewards"] -= 1
          pass
        else: 
          dataset["rewards"] = (dataset["rewards"] - 0.5) * 4
      else:
        min_r, max_r = np.min(dataset["rewards"]), np.max(dataset["rewards"])
        dataset["rewards"] = (dataset["rewards"] - min_r) / (max_r - min_r)
        # dataset["rewards"] = (dataset["rewards"] - 0.5) * 2

    # set sampler
    dataset = Dataset(dataset)

    if self._cfgs.oper:
      if self._cfgs.priority=='return':
        dist = dataset._data['returns']
        dist = (dist - dist.min()) / (dist.max() - dist.min()) + BASE_PROB
        probs = dist / dist.sum()
      elif self._cfgs.priority=='adv':
        weight_list = []
        for seed in range(1, 4):
          wp = Path(__file__).absolute().parent.parent / 'weights' / f'{self._cfgs.env}_{seed}.npy'
          res = np.load(wp, allow_pickle=True).item()
          num_iter, bc_eval_steps = res['iter'], res['eval_steps']
          assert ITER <= num_iter
          weight_list.append(res[ITER])
        weight = np.stack(weight_list, axis=0).mean(axis=0)
        weight = weight - weight.min()
        probs = weight / weight.sum()
        size = dataset.size()
        scale = STD / (probs.std() * size)
        probs = scale*(probs - 1/size) + 1/size
        probs = np.maximum(probs, EPS/size)
        probs = probs/probs.sum() # norm to 1 again
      else:
        raise NotImplementedError(f'prioritiy is measured by return or adv. {self._cfgs.priority} is not supported.')
    
      sampler = PrefetchBalancedSampler(
          probs.squeeze(),
          dataset.size(),
          self._cfgs.batch_size,
          n_prefetch=self._cfgs.n_train_step_per_epoch
        )
    else:
      sampler = RandSampler(dataset.size(), self._cfgs.batch_size)
    
    dataset.set_sampler(sampler)

    return dataset, eval_sampler

  def _setup_rlup(self):
    path = Path(__file__).absolute().parent.parent / 'data'
    dataset = RLUPDataset(
      self._cfgs.rl_unplugged_task_class,
      self._cfgs.env,
      str(path),
      batch_size=self._cfgs.batch_size,
      action_clipping=self._cfgs.clip_action,
    )

    env = DM2Gym(dataset.env)
    eval_sampler = TrajSampler(env, max_traj_length=self._cfgs.max_traj_length)

    return dataset, eval_sampler

  def _setup_dataset(self):
    self._obs_mean = 0
    self._obs_std = 1
    self._obs_clip = np.inf

    dataset_type = DATASET_MAP[self._cfgs.dataset]

    if dataset_type == DATASET.D4RL:
      dataset, eval_sampler = self._setup_d4rl()
    elif dataset_type == DATASET.RLUP:
      dataset, eval_sampler = self._setup_rlup()
    else:
      raise NotImplementedError

    self._observation_dim = eval_sampler.env.observation_space.shape[0]
    self._action_dim = eval_sampler.env.action_space.shape[0]
    self._max_action = float(eval_sampler.env.action_space.high[0])

    if self._cfgs.algo_cfg.target_entropy >= 0.0:
      action_space = eval_sampler.env.action_space
      self._cfgs.algo_cfg.target_entropy = -np.prod(action_space.shape).item()

    return dataset, eval_sampler

  def _setup_policy(self):
    if self._algo_type in [ALGO.MISA, ALGO.CRR, ALGO.IQL, ALGO.CQL]:
      policy = TanhGaussianPolicy(
        self._observation_dim,
        self._action_dim,
        self._cfgs.policy_arch,
        self._cfgs.orthogonal_init,
        self._cfgs.policy_log_std_multiplier,
        self._cfgs.policy_log_std_offset,
        use_layer_norm=self._cfgs.use_layer_norm,
      )
    elif self._algo_type == ALGO.TD3:
      policy = DirectMappingPolicy(
        self._observation_dim,
        self._action_dim,
        self._max_action,
        self._cfgs.policy_arch,
        self._cfgs.orthogonal_init,
      )
    else:
      raise NotImplementedError

    return policy

  def _setup_qf(self):
    if self._algo_type in [ALGO.MISA, ALGO.CRR, ALGO.IQL, ALGO.CQL, ALGO.TD3]:
      qf = FullyConnectedQFunction(
        self._observation_dim,
        self._action_dim,
        self._cfgs.qf_arch,
        self._cfgs.orthogonal_init,
        self._cfgs.use_layer_norm,
        self._cfgs.activation,
      )
    else:
      raise NotImplementedError

    return qf

  def _setup_vf(self):
    vf = FullyConnectedVFunction(
      self._observation_dim,
      self._cfgs.qf_arch,
      self._cfgs.orthogonal_init,
      self._cfgs.use_layer_norm,
      self._cfgs.activation,
    )
    return vf
