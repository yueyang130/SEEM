import absl.app
import absl.flags

# import d4rl  # type: ignore
import gym
import numpy as np
import chex

from algos.conservative_sac import ConservativeSAC
from algos.model import (
  FullyConnectedQFunction,
  SamplerPolicyEncoder,
  TanhGaussianPolicy,
  IdentityEncoder,
  ResEncoder,
  ResQFunction,
  ResTanhGaussianPolicy,
)
from data import Dataset, RandSampler, SlidingWindowSampler, RLUPDataset, DM2Gym, BalancedSampler
import data
from utilities.jax_utils import batch_to_jax
from utilities.replay_buffer import get_d4rl_dataset
from utilities.sampler import TrajSampler
from utilities.utils import (
  # SOTALogger,
  WandBLogger,
  Timer,
  define_flags_with_default,
  get_user_flags,
  prefix_metrics,
  set_random_seed,
)
from viskit.logging import logger, setup_logger
from pathlib import Path
from flax import linen as nn
import tqdm

SOTALogger = WandBLogger

FLAGS_DEF = define_flags_with_default(
  env='halfcheetah-medium-v2',
  max_traj_length=1000,
  seed=42,
  save_model=False,
  batch_size=256,
  reward_scale=1.0,
  reward_bias=0.0,
  clip_action=0.999,
  policy_arch='256-256',
  qf_arch='256-256',
  orthogonal_init=False,
  policy_log_std_multiplier=1.0,
  policy_log_std_offset=-1.0,
  n_epochs=2000,
  bc_epochs=0,
  n_train_step_per_epoch=1000,
  eval_period=10,
  eval_n_trajs=5,
  # configs for trining scheme
  sampler='random', # online, random, balanced
  window_size=3000,  # window size for online sampler
  cql=ConservativeSAC.get_default_config(),
  logging=SOTALogger.get_default_config(),
  dataset='d4rl',
  rl_unplugged_task_class = 'control_suite',
  use_resnet=False,
)


def main(argv):
  FLAGS = absl.flags.FLAGS

  variant = get_user_flags(FLAGS, FLAGS_DEF)
  sota_logger = SOTALogger(
    config=FLAGS.logging, variant=variant, env_name=FLAGS.env
  )
  setup_logger(
    variant=variant,
    exp_id=sota_logger.experiment_id,
    seed=FLAGS.seed,
    base_log_dir=FLAGS.logging.output_dir,
    include_exp_prefix_sub_dir=False
  )

  set_random_seed(FLAGS.seed)

  if FLAGS.dataset == 'd4rl':
    eval_sampler = TrajSampler(
      gym.make(FLAGS.env).unwrapped, FLAGS.max_traj_length
    )

    # Build dataset and sampler
    dataset = get_d4rl_dataset(
      eval_sampler.env,
      FLAGS.cql.nstep,
      FLAGS.cql.discount,
    )

    dataset['rewards'
           ] = dataset['rewards'] * FLAGS.reward_scale + FLAGS.reward_bias
    dataset['actions'] = np.clip(
      dataset['actions'], -FLAGS.clip_action, FLAGS.clip_action
    )
    probs = dataset['rewards']
    probs = (probs - probs.min()) / (probs.max() - probs.min())

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
  else:
    raise NotImplementedError

  observation_dim = eval_sampler.env.observation_space.shape[0]
  action_dim = eval_sampler.env.action_space.shape[0]

  if not FLAGS.use_resnet:
    policy = TanhGaussianPolicy(
      observation_dim, action_dim, FLAGS.policy_arch, FLAGS.orthogonal_init,
      FLAGS.policy_log_std_multiplier, FLAGS.policy_log_std_offset
    )
    qf = FullyConnectedQFunction(
      observation_dim, action_dim, FLAGS.qf_arch, FLAGS.orthogonal_init
    )
    encoder = IdentityEncoder(1, 1024, nn.relu)
  else:
    hidden_dim = FLAGS.cql.res_hidden_size
    encoder_blocks = FLAGS.cql.encoder_blocks
    encoder = ResEncoder(
      encoder_blocks,
      hidden_dim,
      nn.elu,
    )
    policy = ResTanhGaussianPolicy(
      observation_dim,
      action_dim,
      f"{FLAGS.cql.res_hidden_size}",
      FLAGS.orthogonal_init,
      FLAGS.policy_log_std_multiplier,
      FLAGS.policy_log_std_offset,
      FLAGS.cql.head_blocks,
    )
    qf = ResQFunction(
      observation_dim,
      action_dim,
      f"{FLAGS.cql.res_hidden_size}",
      FLAGS.orthogonal_init,
      FLAGS.cql.head_blocks,
    )

  if FLAGS.cql.target_entropy >= 0.0:
    FLAGS.cql.target_entropy = -np.prod(eval_sampler.env.action_space.shape
                                       ).item()

  sac = ConservativeSAC(FLAGS.cql, encoder, policy, qf)
  sampler_policy = SamplerPolicyEncoder(sac, sac.train_params)

  viskit_metrics = {}
  for epoch in range(FLAGS.n_epochs):
    metrics = {'epoch': epoch}

    with Timer() as train_timer:
      for _ in tqdm.tqdm(range(FLAGS.n_train_step_per_epoch)):
        batch = batch_to_jax(dataset.sample())
        metrics.update(
          prefix_metrics(sac.train(batch, bc=epoch < FLAGS.bc_epochs), 'sac')
        )

    with Timer() as eval_timer:
      if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
        trajs = eval_sampler.sample(
          sampler_policy.update_params(sac.train_params),
          FLAGS.eval_n_trajs,
          deterministic=True
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
        if FLAGS.save_model:
          save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
          sota_logger.save_pickle(save_data, 'model.pkl')

    metrics['train_time'] = train_timer()
    metrics['eval_time'] = eval_timer()
    metrics['epoch_time'] = train_timer() + eval_timer()
    sota_logger.log(metrics)
    viskit_metrics.update(metrics)
    logger.record_dict(viskit_metrics)
    logger.dump_tabular(with_prefix=False, with_timestamp=False)

  if FLAGS.save_model:
    save_data = {'sac': sac, 'variant': variant, 'epoch': epoch}
    sota_logger.save_pickle(save_data, 'model.pkl')


if __name__ == '__main__':
  absl.app.run(main)
