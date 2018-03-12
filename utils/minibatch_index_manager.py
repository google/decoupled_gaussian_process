# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A class for generating indices for minibatches."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np


class MinibatchIndexManager(object):
  """Class for managing indices for minibatches of data."""

  def __init__(self,
               num_data_points,
               minibatch_size=None,
               is_sequential=True,
               seed=0):
    """Prepares minibatch settings.

    Args:
      num_data_points: number of data.
      minibatch_size: size of minibatch.
      is_sequential: bool, indicates whether samples will be drawn randomly or
        consecutively after shuffling.
      seed: random seed.
    Raises:
      ValueError: num_data_points is negative.
    """
    self.is_sequential = is_sequential
    self.minibatch_size = minibatch_size
    if num_data_points < 0:
      raise ValueError('Number of data points is negative.')
    self.num_data_points = num_data_points
    self._rng = np.random.RandomState(seed)
    self._idx_all = self._rng.permutation(num_data_points)
    self._counter = 0

  def get_next_minibatch_idx(self, with_counter_increased=True):
    """Retrieves the next minibatch of training data.

    Args:
      with_counter_increased: whether the counter should move after
        retrieveing the data.
    Returns:
      indices for next minibatch.
    """
    if self.minibatch_size is None:
      return self._idx_all
    if not self.is_sequential:
      return self._rng.randint(0, self.num_data_points, self.minibatch_size)
    else:
      begin = self._counter
      end = self._counter + self.minibatch_size
      idx = self._idx_all[np.arange(begin, end) % self.num_data_points]
      if with_counter_increased:
        self._counter = end
        if self._counter >= self.num_data_points:
          logging.info('One epic ends. Shuffle')
          self._counter = 0
          self._idx_all = self._rng.permutation(self.num_data_points)
      return idx
