"""Offline RL with Jax."""

__version__ = "0.0.1"

from algos.conservative_sac import ConservativeSAC
from algos.crr import CRR
from algos.mpo import MPO
from algos.td3 import TD3

__all__ = ['ConservativeSAC', 'CRR', 'MPO', 'TD3']
