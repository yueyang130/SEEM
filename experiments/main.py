import absl.app
import absl.flags

import gym
import numpy as np
import chex

import algos
from algos.model import (
  DecoupledQFunction,
  FullyConnectedNetwork,
  FullyConnectedQFunction,
  FullyConnectedVFunction,
  ResTanhGaussianPolicy,
  SamplerPolicyEncoder,
  ClipGaussianPolicy,
  TanhGaussianPolicy,
  IdentityEncoder,
  ResEncoder,
  ResQFunction,
  ResClipGaussianPolicy,
)
from data import Dataset, RandSampler, SlidingWindowSampler, RLUPDataset, DM2Gym, BalancedSampler
from utilities.jax_utils import batch_to_jax
from utilities.replay_buffer import get_d4rl_dataset, ReplayBuffer
from utilities.sampler import TrajSampler, StepSampler
from utilities.utils import (
  # SOTALogger,
  WandBLogger,
  Timer,
  define_flags_with_default,
  get_user_flags,
  prefix_metrics,
  set_random_seed,
  normalize,
  norm_obs
)
from viskit.logging import logger, setup_logger
from pathlib import Path
from flax import linen as nn
import tqdm

SOTALogger = WandBLogger

FLAGS_DEF = define_flags_with_default(
  env='walker2d-medium-v2',
  algo='IQL',
  max_traj_length=1000,
  seed=42,
  save_model=False,
  batch_size=256,
  reward_scale=1,
  reward_bias=0,
  clip_action=0.999,
  encoder_arch='64-64',
  policy_arch='256-256',
  qf_arch='256-256',
  orthogonal_init=False,
  policy_log_std_multiplier=1.0,
  policy_log_std_offset=-1.0,
  # n_epochs=2000,
  n_epochs=1200,
  bc_epochs=0,
  n_train_step_per_epoch=1000,
  eval_period=10,
  eval_n_trajs=5,
  # configs for trining scheme
  sampler='random', # online, random, balanced
  window_size=3000,  # window size for online sampler
  logging=SOTALogger.get_default_config(),
  dataset='d4rl',
  rl_unplugged_task_class = 'control_suite',
  use_resnet=False,
  use_layer_norm=False,
  activation='elu',
  replay_buffer_size=1000000,
  n_env_steps_per_epoch=1000,
  online=False,
  normalize_reward=False,
  embedding_dim=64,
  decoupled_q=False,
  ibal=True,
  cql_n_actions=50,
  bc_weight_ibal=0.5
)


def main(argv):
  FLAGS = absl.flags.FLAGS
  off_algo = getattr(algos, FLAGS.algo)
  algo_cfg = off_algo.get_default_config()

  algo_cfg['ibal'] = FLAGS.ibal
  algo_cfg['cql_n_actions'] = FLAGS.cql_n_actions
  algo_cfg['bc_weight_ibal'] = FLAGS.bc_weight_ibal

  variant = get_user_flags(FLAGS, FLAGS_DEF)
  for k, v in algo_cfg.items():
    variant[f'algo.{k}'] = v
  
  logging_configs = FLAGS.logging

  is_adroit = any([w in FLAGS.env for w in ['pen', 'hammer', 'door', 'relocate']])
  is_kitchen = 'kitchen' in FLAGS.env
  is_mujoco = any([w in FLAGS.env for w in ['hopper', 'walker', 'cheetah']])
  is_antmaze = 'ant' in FLAGS.env

  env_log_name = ''
  if is_adroit:
    env_log_name = 'Adroit'
  elif is_kitchen:
    env_log_name = 'Kitchen'
  elif is_mujoco:
    env_log_name = 'Mujoco'
  elif is_antmaze:
    env_log_name = 'AntMaze'
  else:
    raise NotImplementedError

  logging_configs['project'] = f"{FLAGS.algo}-{env_log_name}"
  sota_logger = SOTALogger(
    config=logging_configs, variant=variant, env_name=FLAGS.env
  )
  setup_logger(
    variant=variant,
    exp_id=sota_logger.experiment_id,
    seed=FLAGS.seed,
    base_log_dir=FLAGS.logging.output_dir,
    include_exp_prefix_sub_dir=False
  )

  set_random_seed(FLAGS.seed)
  obs_mean = 0
  obs_std = 1
  obs_clip = np.inf

  if FLAGS.dataset == 'd4rl':
    eval_sampler = TrajSampler(
      gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length
    )
    if FLAGS.online:
      train_sampler = StepSampler(
        gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length
      )
    else:
      # Build dataset and sampler
      dataset = get_d4rl_dataset(
        eval_sampler.env,
        algo_cfg.nstep,
        algo_cfg.discount,
      )

      dataset['rewards'
             ] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
      dataset['actions'] = np.clip(
        dataset['actions'], -FLAGS.clip_action, FLAGS.clip_action
      )
      probs = dataset['rewards']
      probs = (probs - probs.min()) / (probs.max() - probs.min())

      if is_antmaze:
        dataset['rewards'] = dataset['rewards'] - 1
      elif is_kitchen or is_adroit:
        obs_mean = dataset['observations'].mean()
        obs_std = dataset['observations'].std()
        obs_clip = 10
        norm_obs(dataset, obs_mean, obs_std, obs_clip)

        dataset['rewards'] = (dataset['rewards'] - np.min(dataset['rewards'])) / np.max(dataset['rewards'])
        dataset['rewards'] = (dataset['rewards'] - 0.5) * 2
      elif FLAGS.normalize_reward and is_mujoco:
        normalize(dataset)
      else:
        print("No reward normalization performed")

      dataset = Dataset(dataset)
      if FLAGS.sampler == 'online':
        sampler = SlidingWindowSampler(
          dataset.size(),
          FLAGS.n_epochs * FLAGS.n_train_step_per_epoch,
          FLAGS.window_size,
          FLAGS.batch_size,
        )
      elif FLAGS.sampler == 'balanced':
        sampler = BalancedSampler(
          probs,
          dataset.size(),
          FLAGS.batch_size,
        )
      else:
        sampler = RandSampler(dataset.size(), FLAGS.batch_size)
      dataset.set_sampler(sampler)

  elif FLAGS.dataset == 'rl_unplugged':
    path = Path(__file__).absolute().parent.parent / 'data'
    dataset = RLUPDataset(
      FLAGS.rl_unplugged_task_class,
      FLAGS.env,
      str(path),
      batch_size=FLAGS.batch_size,
      action_clipping=FLAGS.clip_action,
    )

    env = DM2Gym(dataset.env)
    eval_sampler = TrajSampler(
      env, max_traj_length=FLAGS.max_traj_length
    )
    if FLAGS.online:
      train_sampler = StepSampler(
        env, max_traj_length=FLAGS.max_traj_length
      )
  else:
    raise NotImplementedError

  if FLAGS.online:
    replay_buffer = ReplayBuffer(FLAGS.replay_buffer_size)

  observation_dim = eval_sampler.env.observation_space.shape[0]
  action_dim = eval_sampler.env.action_space.shape[0]

  if is_antmaze or is_kitchen or is_adroit:
    policy_arch = '256-256-256'
    qf_arch = '256-256-256'
  else:
    policy_arch = FLAGS.policy_arch
    qf_arch = FLAGS.qf_arch

  if not FLAGS.use_resnet:
    use_layer_norm = FLAGS.use_layer_norm
    embedding_dim = observation_dim
    encoder = IdentityEncoder(1, embedding_dim, nn.relu)
    if FLAGS.decoupled_q:
      embedding_dim = FLAGS.embedding_dim
      encoder = FullyConnectedNetwork(
        embedding_dim, FLAGS.encoder_arch, FLAGS.orthogonal_init,
        use_layer_norm, 'relu'
      )
    if FLAGS.algo == 'MPO':
      use_layer_norm = True
      policy = ClipGaussianPolicy(
        observation_dim, embedding_dim, action_dim, policy_arch, FLAGS.orthogonal_init,
        FLAGS.policy_log_std_multiplier, FLAGS.policy_log_std_offset, use_layer_norm,
        FLAGS.activation,
      )
    else:
      policy = TanhGaussianPolicy(
        observation_dim, embedding_dim, action_dim, policy_arch, FLAGS.orthogonal_init,
        FLAGS.policy_log_std_multiplier, FLAGS.policy_log_std_offset,
      )

    qf = FullyConnectedQFunction(
      observation_dim,
      action_dim,
      qf_arch,
      FLAGS.orthogonal_init,
      use_layer_norm,
      FLAGS.activation,
    )
    if FLAGS.decoupled_q:
      qf = DecoupledQFunction(
        FLAGS.embedding_dim,
        observation_dim,
        action_dim,
        qf_arch,
        FLAGS.orthogonal_init,
        use_layer_norm,
        FLAGS.activation,
      )
    vf = None
    if FLAGS.algo == 'IQL':
      vf = FullyConnectedVFunction(
        observation_dim,
        qf_arch,
        FLAGS.orthogonal_init,
        use_layer_norm,
        FLAGS.activation,
      )
  else:
    hidden_dim = algo_cfg.res_hidden_size
    encoder_blocks = algo_cfg.encoder_blocks
    encoder = ResEncoder(
      encoder_blocks,
      hidden_dim,
      nn.elu,
    )
    if FLAGS.algo == 'MPO':
      policy = ResClipGaussianPolicy(
        observation_dim,
        action_dim,
        f"{algo_cfg.res_hidden_size}",
        FLAGS.orthogonal_init,
        FLAGS.policy_log_std_multiplier,
        FLAGS.policy_log_std_offset,
        algo_cfg.head_blocks,
      )
    if FLAGS.also == 'IQL':
      raise NotImplementedError
    else:
      policy = ResTanhGaussianPolicy(
        observation_dim,
        action_dim,
        f"{algo_cfg.res_hidden_size}",
        FLAGS.orthogonal_init,
        FLAGS.policy_log_std_multiplier,
        FLAGS.policy_log_std_offset,
        algo_cfg.head_blocks,
      )
    qf = ResQFunction(
      observation_dim,
      action_dim,
      f"{algo_cfg.res_hidden_size}",
      FLAGS.orthogonal_init,
      algo_cfg.head_blocks,
    )

  if algo_cfg.target_entropy >= 0.0:
    algo_cfg.target_entropy = -np.prod(eval_sampler.env.action_space.shape
                                       ).item()

  if FLAGS.algo == 'IQL':
    agent = off_algo(algo_cfg, encoder, policy, qf, vf)
  else:
    agent = off_algo(algo_cfg, encoder, policy, qf, decoupled_q=FLAGS.decoupled_q)
  sampler_policy = SamplerPolicyEncoder(agent, agent.train_params)

  viskit_metrics = {}
  returns = []
  normalized_returns = []
  for epoch in range(FLAGS.n_epochs):
    metrics = {'epoch': epoch}

    if FLAGS.online:
      with Timer() as rollout_timer:
        train_sampler.sample(
          sampler_policy.update_params(agent.train_params),
          FLAGS.n_env_steps_per_epoch, deterministic=False,
          replay_buffer=replay_buffer
        )
        metrics['env_steps'] = replay_buffer.total_steps
        metrics['epoch'] = epoch

    with Timer() as train_timer:
      for _ in tqdm.tqdm(range(FLAGS.n_train_step_per_epoch)):
        if FLAGS.online:
          batch = batch_to_jax(replay_buffer.sample(FLAGS.batch_size))
        else:
          batch = batch_to_jax(dataset.sample())
        metrics.update(
          prefix_metrics(agent.train(batch, bc=epoch < FLAGS.bc_epochs), 'agent')
        )

    with Timer() as eval_timer:
      if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
        trajs = eval_sampler.sample(
          sampler_policy.update_params(agent.train_params),
          FLAGS.eval_n_trajs,
          deterministic=True,
          obs_statistics=(obs_mean, obs_std, obs_clip)
        )

        metrics['average_return'] = np.mean(
          [np.sum(t['rewards']) for t in trajs]
        )
        metrics['average_traj_length'] = np.mean(
          [len(t['rewards']) for t in trajs]
        )
        metrics['average_normalizd_return'] = np.mean(
          [
            eval_sampler.env.get_normalized_score(np.sum(t['rewards']))
            for t in trajs
          ]
        )
  
        # get the average of the final 10 evaluations as performance
        returns.append(metrics['average_return'])
        normalized_returns.append(metrics['average_normalizd_return'])
        metrics['average_10_return'] = np.mean(
          returns[-min(10, len(returns)):]
        )
        metrics['average_10_normalizd_return'] = np.mean(
          normalized_returns[-min(10, len(returns)):]
        )

        if FLAGS.save_model:
          save_data = {'agent': agent, 'variant': variant, 'epoch': epoch}
          sota_logger.save_pickle(save_data, f'model_{epoch}.pkl')

    if FLAGS.online:
      metrics['rollout_time'] = rollout_timer()
    metrics['train_time'] = train_timer()
    metrics['eval_time'] = eval_timer()
    metrics['epoch_time'] = train_timer() + eval_timer()
    sota_logger.log(metrics)
    viskit_metrics.update(metrics)
    logger.record_dict(viskit_metrics)
    logger.dump_tabular(with_prefix=False, with_timestamp=False)

  if FLAGS.save_model:
    save_data = {'agent': agent, 'variant': variant, 'epoch': epoch}
    sota_logger.save_pickle(save_data, 'model_final.pkl')


if __name__ == '__main__':
  absl.app.run(main)
