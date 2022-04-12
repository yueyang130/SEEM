import absl.app
import absl.flags

# import d4rl  # type: ignore
import gym
import numpy as np

from algos.conservative_sac import ConservativeSAC
from algos.model import (
  FullyConnectedQFunction,
  SamplerPolicy,
  TanhGaussianPolicy,
)
from data import Dataset, RandSampler, SlidingWindowSampler
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
  online=False,  # use online training
  window_size=3000,  # window size for online sampler
  cql=ConservativeSAC.get_default_config(),
  logging=SOTALogger.get_default_config(),
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
  dataset = Dataset(dataset)
  if FLAGS.online:
    sampler = SlidingWindowSampler(
      dataset.size(),
      FLAGS.n_epochs * FLAGS.n_train_step_per_epoch,
      FLAGS.window_size,
      FLAGS.batch_size,
    )
  else:
    sampler = RandSampler(dataset.size(), FLAGS.batch_size)
  dataset.set_sampler(sampler)

  observation_dim = eval_sampler.env.observation_space.shape[0]
  action_dim = eval_sampler.env.action_space.shape[0]

  policy = TanhGaussianPolicy(
    observation_dim, action_dim, FLAGS.policy_arch, FLAGS.orthogonal_init,
    FLAGS.policy_log_std_multiplier, FLAGS.policy_log_std_offset
  )
  qf = FullyConnectedQFunction(
    observation_dim, action_dim, FLAGS.qf_arch, FLAGS.orthogonal_init
  )

  if FLAGS.cql.target_entropy >= 0.0:
    FLAGS.cql.target_entropy = -np.prod(eval_sampler.env.action_space.shape
                                       ).item()

  sac = ConservativeSAC(FLAGS.cql, policy, qf)
  sampler_policy = SamplerPolicy(sac.policy, sac.train_params['policy'])

  viskit_metrics = {}
  for epoch in range(FLAGS.n_epochs):
    metrics = {'epoch': epoch}

    with Timer() as train_timer:
      for _ in range(FLAGS.n_train_step_per_epoch):
        # batch = batch_to_jax(subsample_batch(dataset, FLAGS.batch_size))
        print(_)
        batch = batch_to_jax(dataset.sample())
        metrics.update(
          prefix_metrics(sac.train(batch, bc=epoch < FLAGS.bc_epochs), 'sac')
        )

    with Timer() as eval_timer:
      if epoch == 0 or (epoch + 1) % FLAGS.eval_period == 0:
        trajs = eval_sampler.sample(
          sampler_policy.update_params(sac.train_params['policy']),
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
