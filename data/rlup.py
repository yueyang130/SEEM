import rlds
import tensorflow_datasets as tfds
import tensorflow as tf


def traj_fn(traj_length, obs_keys):
  def step_proc_fn(batch):
    obs = tf.concat(
      [batch[rlds.OBSERVATION][k] for k in obs_keys], axis=-1
    )
    return {
      rlds.OBSERVATION: obs,
      rlds.REWARD: batch[rlds.REWARD],
      rlds.ACTION: batch[rlds.ACTION],
      rlds.IS_FIRST: batch[rlds.IS_FIRST],
      rlds.IS_LAST: batch[rlds.IS_LAST],
    }

  def make_traj_ds(episode):
    step_data = episode[rlds.STEPS]
    start = tf.random.uniform(shape=(), minval=0, maxval=traj_length, dtype=tf.int64)
    step_data = step_data.map(step_proc_fn).skip(start)
    trajectory = step_data.batch(traj_length, drop_remainder=True)
    return trajectory
  
  return make_traj_ds


class OfflineDataset:
  def __init__(self, domain='rlu_control_suite', task='walker_walk', batch_size=256, episode_shuffle_size=10, traj_length=10, shuffle_num_steps=50000) -> None:
    self._domain = domain
    self._task = task
    self._obs_keys = []
    if 'control_suite' in self._domain:
      self._obs_keys.extend(
        ['height', 'orientations', 'velocity']
      )
    else:
      raise NotImplementedError

    self._ds_name = f"{domain}/{task}"
    self._bs = batch_size

    _ds = tfds.load(self._ds_name)['train']
    _ds = _ds.shuffle(episode_shuffle_size).interleave(
      traj_fn(traj_length, self._obs_keys),
      cycle_length=100,
      block_length=1,
      deterministic=False,
      num_parallel_calls=tf.data.AUTOTUNE
    )
    _ds = _ds.shuffle(
      shuffle_num_steps // traj_length,
      reshuffle_each_iteration=True
    )
    _ds = _ds.batch(batch_size)
    self._ds = iter(_ds)
  
  def sample(self):
    # data has shape [B, T, H]
    return tfds.as_numpy(next(self._ds))


class TransitionDataset(OfflineDataset):
  def __init__(self, domain='rlu_control_suite', task='walker_walk', batch_size=256, episode_shuffle_size=10, shuffle_num_steps=50000) -> None:
      super().__init__(domain, task, batch_size, episode_shuffle_size, 2, shuffle_num_steps)
  
  def sample(self):
    seq_data = super().sample()
    print(seq_data)


if __name__ == "__main__":
  off_ds = TransitionDataset()
  sampled_data = off_ds.sample()
