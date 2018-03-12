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
"""Utility functions and classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# The constant for a `small` number used on algorithms.
EPSILON = 1e-5

# The implementation of algorithms support both float32 and float64.
tf_float = tf.float32
tf_int = tf.int32
np_float = np.float32


def build_squared_distance(xa, xb, matrix=None, diagonal=None,
                           is_inverse=False):
  """Build a matrix for pairwise distances.

  Args:
    xa: rank-2 tensor.
    xb: rank-2 tensor.
    matrix: the SYMMETRIC matrix in computing the distance.
    diagonal: rank-1 tensor.
    is_inverse: bool, indicates whether the matrix `M` is inversed.
  Returns:
    rank-2 tensor, with ijth item being ((u - v)^T M D (u - v)), where u, v are
    the ith and jth row of `xa` and `xb` respectively, and M is `matrix`, and D
    is `diagonal`.
  """

  if diagonal is None:
    diagonal = 1.0
  xa_mat, xb_mat = xa * diagonal, xb * diagonal
  if matrix is not None:
    if not is_inverse:
      xa_mat, xb_mat = tf.matmul(xa_mat, matrix), tf.matmul(xb_mat, matrix)
    else:
      xa_mat = tf.transpose(tf.matrix_solve(matrix, tf.transpose(xa_mat)))
      xb_mat = tf.transpose(tf.matrix_solve(matrix, tf.transpose(xb_mat)))
  xa_squared_sum = tf.reduce_sum(xa_mat * xa, axis=1)
  xb_squared_sum = tf.reduce_sum(xb_mat * xb, axis=1)
  cross_terms = -2.0 * tf.matmul(xa_mat, xb, transpose_b=True)
  squared_distances = (
      tf.reshape(xa_squared_sum,
                 (-1, 1)) + cross_terms + tf.reshape(xb_squared_sum, (1, -1)))
  return squared_distances


def build_learning_rate_with_decay(r0, r1, global_step):
  """Builds a tensor for learning rate.

  Learning rate: r_t = r0 (1 + r1 sqrt(t))^{-1}.

  Args:
    r0: scalar.
    r1: scalar.
    global_step: rank-0 tensor.
  Returns:
    rank-0 tensor for learning rate.
  """
  sqrt_t = tf.sqrt(tf.cast(global_step, dtype=tf_float))
  return r0 / (1.0 + r1 * sqrt_t)


def init_hyperparameters(x,
                         y,
                         percentile=50.0,
                         num_data_points_for_pairwise_distance=None,
                         random_state=None):
  """Initializes the hyperparameters based on heuristics.

  Parameters for auto relevance determination squared exponentation kernel,
  and Gaussian likelihood.

  Args:
    x: rank-2 array, inputs or features for computing pairwise distances.
    y: rank-2 array, column, labels or targets for computing variance.
    percentile: in interval [0, 100], the percentile of pairwise distances of
      separate dimensions to initialize length_scale.
    num_data_points_for_pairwise_distance: number of randomly picked data
      points to compute pairwise distances for kernel length scale. `None` to
      use all.
    random_state: np RandomState.
  Returns:
    length_scale (rank-1 array), signal_stddev (scalar),
    likelihood_variance (scalar).
  """
  dim_x = x.shape[1]

  # Length scale.
  if num_data_points_for_pairwise_distance is None:
    idx = range(x.shape[0])
  else:
    if random_state is None:
      random_state = np.random.RandomState()
    idx = random_state.choice(x.shape[0], num_data_points_for_pairwise_distance)
  sampled_x = x[idx]
  length_scale = np.zeros(dim_x)
  for i in range(dim_x):
    distances = np.abs(sampled_x[:, [i]] - sampled_x[:, i]).flatten()
    length_scale[i] = np.sqrt(dim_x) * np.percentile(
        distances, percentile, interpolation='nearest')
  # Signal
  signal_stddev = np.std(y)
  likelihood_variance = np.var(y) / 100.0

  return length_scale, signal_stddev, likelihood_variance
