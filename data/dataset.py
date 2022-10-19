# Copyright 2022 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset utils."""
import d4rl
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
