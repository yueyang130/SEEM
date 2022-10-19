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
  ConservativeSAC = 2
  CRR = 3
  MPO = 4
  TD3 = 5
  IQL = 6

ALGO_MAP = {
  'MISA': ALGO.MISA,
  'ConservativeSAC': ALGO.ConservativeSAC,
  'CRR': ALGO.CRR,
  'MPO': ALGO.MPO,
  'TD3': ALGO.TD3,
  'IQL': ALGO.IQL
}
