"""To be cleaned later."""

import collections

import d4rl
import gym
import numpy as np
from tqdm import tqdm

Batch = collections.namedtuple(
  "Batch",
  ["observations", "actions", "rewards", "masks", "next_observations"]
)


def split_into_trajectories(
  observations, actions, rewards, masks, dones_float, next_observations
):
  trajs = [[]]

  for i in tqdm(range(len(observations))):
    trajs[-1].append(
      (
        observations[i],
        actions[i],
        rewards[i],
        masks[i],
        dones_float[i],
        next_observations[i],
      )
    )
    if dones_float[i] == 1.0 and i + 1 < len(observations):
      trajs.append([])

  return trajs


class Dataset(object):

  def __init__(
    self,
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    masks: np.ndarray,
    dones_float: np.ndarray,
    next_observations: np.ndarray,
    size: int,
  ):

    self.observations = observations
    self.actions = actions
    self.rewards = rewards
    self.masks = masks
    self.dones_float = dones_float
    self.next_observations = next_observations
    self.size = size

  def sample(self, batch_size: int) -> Batch:
    indx = np.random.randint(self.size, size=batch_size)
    return Batch(
      observations=self.observations[indx],
      actions=self.actions[indx],
      rewards=self.rewards[indx],
      masks=self.masks[indx],
      next_observations=self.next_observations[indx],
    )


class D4RLDataset(Dataset):

  def __init__(
    self, env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5
  ):
    self.raw_dataset = dataset = d4rl.qlearning_dataset(env)

    if clip_to_eps:
      lim = 1 - eps
      dataset["actions"] = np.clip(dataset["actions"], -lim, lim)

    dones_float = np.zeros_like(dataset["rewards"])

    for i in range(len(dones_float) - 1):
      if (
        np.linalg.
        norm(dataset["observations"][i + 1] - dataset["next_observations"][i])
        > 1e-6 or dataset["terminals"][i] == 1.0
      ):
        dones_float[i] = 1
      else:
        dones_float[i] = 0

    dones_float[-1] = 1

    super().__init__(
      dataset["observations"].astype(np.float32),
      actions=dataset["actions"].astype(np.float32),
      rewards=dataset["rewards"].astype(np.float32),
      masks=1.0 - dataset["terminals"].astype(np.float32),
      dones_float=dones_float.astype(np.float32),
      next_observations=dataset["next_observations"].astype(np.float32),
      size=len(dataset["observations"]),
    )


def get_traj_dataset(env, sorting=True):
  env = gym.make(env) if isinstance(env, str) else env
  dataset = D4RLDataset(env)
  trajs = split_into_trajectories(
    dataset.observations, dataset.actions, dataset.rewards, dataset.masks,
    dataset.dones_float, dataset.next_observations
  )

  def compute_returns(traj):
    episode_return = 0
    for _, _, rew, _, _, _ in traj:
      episode_return += rew

    return episode_return

  if sorting:
    trajs.sort(key=compute_returns)

  # NOTE: this raw_dataset is not sorted
  return trajs, dataset.raw_dataset


def nstep_reward_prefix(rewards, nstep=5, gamma=0.9):
  gammas = np.array([gamma**i for i in range(nstep)])
  nstep_rewards = np.convolve(rewards, gammas)[nstep - 1:]
  return nstep_rewards


def get_nstep_dataset(env, nstep=5, gamma=0.9, sorting=True):
  gammas = np.array([gamma**i for i in range(nstep)])
  trajs, raw_dataset = get_traj_dataset(env, sorting)

  obss, acts, terms, next_obss, nstep_rews = [], [], [], [], []
  for traj in trajs:
    L = len(traj)
    rewards = np.array([ts[2] for ts in traj])
    cum_rewards = np.convolve(rewards, gammas)[nstep - 1:]
    nstep_rews.append(cum_rewards)
    next_obss.extend([traj[min(i + nstep - 1, L - 1)][-1] for i in range(L)])
    obss.extend([traj[i][0] for i in range(L)])
    acts.extend([traj[i][1] for i in range(L)])
    terms.extend([bool(1 - traj[i][3]) for i in range(L)])

  dataset = {}
  dataset['observations'] = np.stack(obss)
  dataset['actions'] = np.stack(acts)
  dataset['next_observations'] = np.stack(next_obss)
  dataset['rewards'] = np.concatenate(nstep_rews)
  dataset['terminals'] = np.stack(terms)

  assert len(dataset['rewards']) == len(raw_dataset['rewards'])
  assert dataset['next_observations'].shape == raw_dataset['next_observations'
                                                          ].shape

  return dataset


if __name__ == '__main__':
  env_name = 'halfcheetah-medium-v0'
  # trajs = get_traj_dataset(env_name)
  dataset = get_nstep_dataset(env_name)
  import pdb
  pdb.set_trace()
  print('hello world')