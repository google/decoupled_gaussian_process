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
"""A class for processing and supplying data in numpy array format.

Feautures include, for example, generating random data, whitening data, sampling
minibatch of data, and computing hyperparameters based on heuristics.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from decoupled_gaussian_process import utils
from decoupled_gaussian_process.utils import minibatch_index_manager
import numpy as np
import sklearn.preprocessing
import tensorflow as tf

Dataset = collections.namedtuple('Dataset', ['training', 'test'])
DataPoints = collections.namedtuple('DataPoints', ['x', 'y'])
NormalizationOps = collections.namedtuple(
    'NormalizationOps',
    ['normalize_x', 'inv_normalize_x', 'normalize_y', 'inv_normalize_y'])


class NumpyDataInterface(object):
  """Data related stuff.

  This class has functions for processing data, for example, generating random
  data, whitening data, sampling minibatch of data, and computing
  hyperparameters based on heuristics.
  """

  def __init__(self, seed=0):
    self.data = Dataset(DataPoints(None, None), DataPoints(None, None))
    self.nomalized_data = Dataset(
        DataPoints(None, None), DataPoints(None, None))
    self.normalization_ops = NormalizationOps(None, None, None, None)
    self._rng = np.random.RandomState(seed)
    self._index_manager = None

  def generate_data_with_random_uniform_extra_dimensions(
      self,
      func,
      num_training_data_points,
      num_test_data_points,
      noise_stddev,
      x_limit,
      dim_extra=0):
    """Generates data.

    Generates data with only the first dimension following `func`, other
      dimensions are uniformly distributed noise.

    Args:
      func: function for the first dimension.
      num_training_data_points: number of training samples.
      num_test_data_points: number of test samples.
      noise_stddev: standard deviation of Gaussian noise in training data.
      x_limit: the domain of training data and the range of noisy dimensions.
      dim_extra: extra noisy dimensions.
    Returns:
      `Dataset` namedtuple.
    """
    x_training = self._rng.uniform(x_limit[0], x_limit[1],
                                   (num_training_data_points, 1 + dim_extra))
    y_training = (
        func(x_training[:, [0]]) +
        self._rng.randn(num_training_data_points, 1) * noise_stddev)
    x_test = self._rng.uniform(x_limit[0], x_limit[1],
                               (num_test_data_points, 1 + dim_extra))
    y_test = func(x_test[:, [0]])
    return Dataset(
        training=DataPoints(x=x_training, y=y_training),
        test=DataPoints(x=x_test, y=y_test))

  def prepare_data_for_minibatch(self,
                                 data,
                                 minibatch_size=None,
                                 is_sequential=True,
                                 with_whitening=False,
                                 seed=0):
    """Prepare data with whitening for minibatch.

    Args:
      data: `Dataset` namedtuple.
      minibatch_size: size of minibatch.
      is_sequential: bool, indicates whether samples will be drawn randomly or
        consecutively after shuffling.
      with_whitening: bool, indicates whether data should be whitened.
      seed: random seed.
    """
    # Whitening or normalization.
    if with_whitening:
      x_scalar = sklearn.preprocessing.StandardScaler().fit(data.training.x)
      y_scalar = sklearn.preprocessing.StandardScaler().fit(data.training.y)
      self.normalization_ops = NormalizationOps(
          normalize_x=lambda x: x_scalar.transform(x, copy=True),
          inv_normalize_x=lambda x: x_scalar.inverse_transform(x, copy=True),
          normalize_y=lambda x: y_scalar.transform(x, copy=True),
          inv_normalize_y=lambda x: y_scalar.inverse_transform(x, copy=True))
    else:
      self.normalization_ops = NormalizationOps(
          normalize_x=lambda x: x,
          inv_normalize_x=lambda x: x,
          normalize_y=lambda x: x,
          inv_normalize_y=lambda x: x)
    self.data = data

    # Normalize all the data in advance.
    training_data_points = DataPoints(
        x=self.normalization_ops.normalize_x(self.data.training.x),
        y=self.normalization_ops.normalize_y(self.data.training.y))
    test_data_points = DataPoints(
        x=self.normalization_ops.normalize_x(self.data.test.x),
        y=self.normalization_ops.normalize_y(self.data.test.y))
    self.normalized_data = Dataset(training_data_points, test_data_points)

    # Setup index manager.
    self._index_manager = minibatch_index_manager.MinibatchIndexManager(
        self.data.training.x.shape[0], minibatch_size, is_sequential, seed)

  def get_next_normalized_training_batch(self, increase_counter=True):
    """Retrieves the next batch of training data."""
    idx = self._index_manager.get_next_minibatch_idx(increase_counter)
    return (self.normalized_data.training.x[idx],
            self.normalized_data.training.y[idx])

  def build_feed_dict(self):
    """Build a feed_dict function.

    The feed_dict function returns a minibatch of training dataset when
    `during_training` is True, and returns the whole test dataset otherwise.

    Returns:
      Tuple, (tf.placeholder for x, tf.placeholder for y, feed_dict function).
    """
    x_placeholder = tf.placeholder(
        utils.tf_float, shape=[None, self.data.training.x.shape[1]])
    y_placeholder = tf.placeholder(utils.tf_float, shape=[None, 1])

    def feed_dict(during_training, idx_y=0):
      if during_training:
        x, y = self.get_next_normalized_training_batch()
      else:
        x = self.normalized_data.test.x
        y = self.normalized_data.test.y
      y = y[:, [idx_y]]
      return {x_placeholder: x, y_placeholder: y}

    return x_placeholder, y_placeholder, feed_dict

  def sample_normalized_training_x(self, num_data_points_to_sample):
    """Sample x from normalized training data."""
    idx = self._rng.randint(0, self.data.training.x.shape[0],
                            num_data_points_to_sample)
    return self.normalized_data.training.x[idx]

  def sample_normalized_training(self, num_data_points_to_sample):
    """Sample from normalized training data."""
    idx = self._rng.randint(0, self.data.training.x.shape[0],
                            num_data_points_to_sample)
    return (self.normalized_data.training.x[idx],
            self.normalized_data.training.y[idx])
