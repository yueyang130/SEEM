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

"""Dataset samplers."""
from typing import Any

import numpy as np

Array = Any


class RandSampler(object):
	"""A random sampler."""

	def __init__(self, max_size: int, batch_size: int = 1) -> None:
		self._max_size = max_size
		self._batch_size = batch_size

	def sample(self):
		"""Return an array of sampled indices."""

		return np.random.randint(self._max_size, size=self._batch_size)


"""
code from https://github.com/sail-sg/OPER/blob/16e0c7479249af13c17a73bcd4a4bd7b1e97077e/utils.py#L15
"""
class PrefetchBalancedSampler(object):
  """A prefetch balanced sampler."""

  def __init__(self, probs: Array, max_size: int, batch_size: int, n_prefetch: int) -> None:
    self._max_size = max_size
    self._batch_size = batch_size
    self.n_prefetch = min(n_prefetch, max_size//batch_size)
    self._probs = probs / np.sum(probs)
    self.cnt = self.n_prefetch - 1
  
  def sample(self):

    self.cnt = (self.cnt+1)%self.n_prefetch
    if self.cnt == 0:
      self.indices = np.random.choice(self._max_size, 
          size=self._batch_size * self.n_prefetch, p=self._probs)
    return self.indices[self.cnt*self._batch_size : (self.cnt+1)*self._batch_size]