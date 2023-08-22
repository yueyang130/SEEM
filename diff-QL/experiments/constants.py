from enum import IntEnum


class ENV(IntEnum):
  Adroit = 1
  Kitchen = 2
  Mujoco = 3
  Antmaze = 4


ENV_MAP = {
  'pen': ENV.Adroit,
  'hammer': ENV.Adroit,
  'door': ENV.Adroit,
  'relocate': ENV.Adroit,
  'kitchen': ENV.Kitchen,
  'hopper': ENV.Mujoco,
  'walker': ENV.Mujoco,
  'cheetah': ENV.Mujoco,
  'finger': ENV.Mujoco,
  'humanoid': ENV.Mujoco,
  'cartpole': ENV.Mujoco,
  'fish': ENV.Mujoco,
  'antmaze': ENV.Antmaze
}

ENVNAME_MAP = {
  ENV.Adroit: 'Adroit',
  ENV.Kitchen: 'Kitchen',
  ENV.Mujoco: 'Mujoco',
  ENV.Antmaze: 'Antmaze'
}


class ALGO(IntEnum):
  MISA = 1
  CQL = 2
  CRR = 3
  MPO = 4
  TD3 = 5
  IQL = 6


ALGO_MAP = {
  'MISA': ALGO.MISA,
  'CQL': ALGO.CQL,
  'CRR': ALGO.CRR,
  'MPO': ALGO.MPO,
  'TD3': ALGO.TD3,
  'IQL': ALGO.IQL
}


class DATASET(IntEnum):
  D4RL = 1
  RLUP = 2


DATASET_MAP = {'d4rl': DATASET.D4RL, 'rl_unplugged': DATASET.RLUP}

DATASET_ABBR_MAP = {'d4rl': 'D4RL', 'rl_unplugged': 'RLUP'}
