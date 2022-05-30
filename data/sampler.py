"""Samplers."""
from dataclasses import replace
import numpy as np
from typing import Any

Array = Any


class RandSampler(object):
  """A random sampler."""

  def __init__(self, max_size: int, batch_size: int = 1) -> None:
    self._max_size = max_size
    self._batch_size = batch_size

  def sample(self):
    """Return an array of sampled indices."""

    return np.random.randint(self._max_size, size=self._batch_size)


class BalancedSampler(object):
  """A balanced sampler."""

  def __init__(self, probs: Array, max_size: int, batch_size: int) -> None:
    self._max_size = max_size
    self._batch_size = batch_size
    self._probs = probs / np.sum(probs)
  
  def sample(self):
    return np.random.choice(self._max_size, size=self._batch_size, replace=False, p=self._probs)

class SlidingWindowSampler(object):
  """A random sampler with sliding windows."""

  def __init__(
    self,
    max_size: int,
    max_iters: int,
    window_size: int,
    batch_size: int = 1,
  ) -> None:
    self._max_size = max_size
    self._max_iters = max_iters
    self._window_size = window_size
    self._batch_size = batch_size
    self._iter = 0

  def get_pointer(self, i):
    """Get the starting point of the sliding window at iteration i."""
    # return int(
    #   i * (self._max_size - self._window_size) / (self._max_iters - 1)
    # )
    return min(
      int(i * self._max_size / (self._max_iters - 1)),
      int(self._max_size - self._window_size)
    )

  def sample(self):
    """Sample a batch of indices in the current sliding window.
    (pointer, pointer + window_size).
    """
    pointer = self.get_pointer(self._iter)
    indices = np.random.randint(self._window_size, size=self._batch_size)
    self._iter += 1
    return indices + pointer


if __name__ == '__main__':
  sampler = SlidingWindowSampler(
    max_size=10000,
    max_iters=108,
    window_size=300,
    batch_size=30,
  )
  for i in range(10):
    print(sampler.get_pointer(i), sampler.sample())
  # import pdb
  # pdb.set_trace()
