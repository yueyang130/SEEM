"""Datasets."""
import numpy as np


class Dataset(object):
  """Dataset."""

  def __init__(self, data: dict) -> None:
    self._data = data
    self._keys = list(data.keys())
    self._sampler = None

  def size(self):
    return len(self._data[self._keys[0]])

  def retrieve(self, indices: np.ndarray):
    "Get a batch of data."
    indexed = {}

    for key in self._keys:
      indexed[key] = self._data[key][indices, ...]
    return indexed

  @property
  def sampler(self):
    assert self._sampler is not None
    return self._sampler

  def set_sampler(self, sampler):
    self._sampler = sampler

  def sample(self):
    assert self._sampler is not None

    indices = self._sampler.sample()
    return self.retrieve(indices)
