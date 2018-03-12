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
"""A class for managing bases in decoupled Gaussian process model.

It manages the bases in the decoupled Gaussian process model so that the number
of bases can be added incrementally in order to ease optimization and add more
flexibility to the model. It is compatible of the usage of optimizers with
memory, such as adam and momentum.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from decoupled_gaussian_process import utils
import numpy as np
import tensorflow as tf


class BasesMananger(object):
  """Manages the optimization procedure for bases."""

  def __init__(self,
               dim_bases,
               num_mean_bases,
               num_cov_bases,
               num_mean_bases_init,
               num_cov_bases_init,
               bases_to_sample=None,
               with_normalization=True,
               normalization_threshold=2048):
    """Initialize.

    Args:
      dim_bases: the dimension of bases.
      num_mean_bases: number of bases for mean.
      num_cov_bases: number of bases for covariance.
      num_mean_bases_init: initial number of bases for mean.
      num_cov_bases_init: initial number of bases for covariance.
      bases_to_sample: rank-2 tensor, bases to sample from to initialize
        bases for mean and covariance.
      with_normalization: bool, indicates whether statistics for inducing points
        will be normalized.
      normalization_threshold: integer, a parameter in normalization, in order
        to make training step less sensitive to number of bases.
    """

    # Make sure the initial size is no larger than final size.
    num_mean_bases_init = min(num_mean_bases, num_mean_bases_init)
    num_cov_bases_init = min(num_cov_bases, num_cov_bases_init)

    # Needed by _build_stats_normalizer()
    self.with_normalization = with_normalization
    self.normalization_threshold = normalization_threshold

    # Bases to initialize all the bases.
    # Make sure the bases for mean and covariance are initialized the same.
    max_num_bases = max(num_mean_bases, num_cov_bases)
    if bases_to_sample is not None:
      bases_init = self._build_bases_sampling(bases_to_sample, max_num_bases)
    else:
      stddev = (
          utils.EPSILON / np.sqrt(max_num_bases * dim_bases).astype(
              utils.np_float))
      bases_init = tf.truncated_normal(
          shape=[max_num_bases, dim_bases], stddev=stddev, dtype=utils.tf_float)

    # Couples mean and covariance bases when their initial numbers are the
    # same.
    is_coupled = num_mean_bases_init == num_cov_bases_init
    self._is_coupled = tf.get_variable(
        'is_coupled', trainable=False, initializer=is_coupled)

    # Mean bases, tf variables.
    with tf.variable_scope('mean'):
      (self._num_mean_bases, self._mean_bases, self._mean_stats,
       self._mean_log_coeff) = self._create_tf_variables(
           num_mean_bases, num_mean_bases_init, bases_init, [num_mean_bases, 1],
           dim_bases)

    # Covariance bases, tf variables.
    with tf.variable_scope('covariance'):
      (self._num_cov_bases, self._cov_bases,
       self._cov_stats, self._cov_log_coeff) = self._create_tf_variables(
           num_cov_bases, num_cov_bases_init, bases_init,
           [num_cov_bases, num_cov_bases], dim_bases)

      # Build non-tf or interfacing variables.
      self._build_public_variables()

  def _create_tf_variables(self, num_bases_max, num_bases_init, bases_init,
                           stats_shape, dim_bases):
    """Create tf variables for mean or covariance operators.

    Args:
      num_bases_max: the upper limit of the number of bases .
      num_bases_init: the initial number of bases.
      bases_init: rank-2 tensor or array, to initialize all the bases.
      stats_shape: list or array, the shape of the stats for the bases.
      dim_bases: dimension of bases.
    Returns:
      tuple of tf variables.
    """
    num_bases = tf.get_variable(
        'num_bases',
        initializer=num_bases_init,
        trainable=False,
        dtype=utils.tf_int)
    bases = tf.get_variable(
        'bases', initializer=bases_init[:num_bases_max], dtype=utils.tf_float)
    stddev = (
        utils.EPSILON / np.sqrt(np.prod(stats_shape)).astype(utils.np_float))
    stats = tf.get_variable(
        'stats',
        initializer=tf.truncated_normal(
            shape=stats_shape, stddev=stddev, dtype=utils.tf_float),
        dtype=utils.tf_float)
    log_coeff = tf.get_variable(
        'log_coeff',
        initializer=tf.truncated_normal(
            shape=[num_bases_max, dim_bases],
            stddev=utils.EPSILON,
            dtype=utils.tf_float),
        dtype=utils.tf_float)
    return num_bases, bases, stats, log_coeff

  def _build_public_variables(self):
    """Build public or idm_controlnterfacing variables."""
    # Mean bases.
    self.mean_bases = self._mean_bases[:self._num_mean_bases]
    self.mean_stats = self._mean_stats[:self._num_mean_bases]
    self.mean_coeff = tf.exp(self._mean_log_coeff[:self._num_mean_bases])

    # Covariance bases.
    self.cov_bases = tf.cond(self._is_coupled, lambda: self.mean_bases,
                             lambda: self._cov_bases[:self._num_cov_bases])
    self.cov_stats = self._cov_stats[:self._num_cov_bases, :self._num_cov_bases]
    self.cov_coeff = tf.cond(
        self._is_coupled, lambda: self.mean_coeff,
        lambda: tf.exp(self._cov_log_coeff[:self._num_cov_bases]))

    # Normalization based on the current number of bases.
    self.mean_stats /= self._build_stats_normalizer(self._num_mean_bases,
                                                    self.mean_stats)
    self.cov_stats /= self._build_stats_normalizer(self._num_cov_bases,
                                                   self.cov_stats)

  def _build_stats_normalizer(self, num_bases, stats):
    """Build normalizer for stats variable."""
    normalizer = tf.constant(1.0, utils.tf_float)
    if self.with_normalization:
      normalizer = tf.maximum(normalizer,
                              tf.cast(num_bases - self.normalization_threshold,
                                      utils.tf_float))
      # Adjust based on the number of elements in stats, so that the change in
      # the norm of stats for mean and covarance are the same.
      # exponent is 0.5 for mean, and 1.0 for covariance.
      exponent = tf.log(
          tf.cast(tf.reduce_prod(tf.shape(stats)), utils.tf_float))
      exponent /= 2.0 * tf.log(tf.cast(tf.shape(stats)[0], utils.tf_float))
      normalizer = tf.pow(normalizer, exponent)
    return normalizer

  def decouple_bases(self, session):
    """Decouples the bases for mean and covariance.

    Keeps the number of bases the same. After decouping, bases for mean and
    cov will be optimized separately.
    Args:
      session: tf session.
    Raises:
      ValueError: raised when the bases are already decoupled.
    """
    if session.run(self._is_coupled):
      # Copy part of mean bases to cov bases.
      session.run(
          tf.scatter_update(self._cov_bases, tf.range(self._num_cov_bases),
                            session.run(
                                self._mean_bases[:self._num_cov_bases])))
      session.run(
          tf.scatter_update(self._cov_log_coeff, tf.range(self._num_cov_bases),
                            session.run(
                                self._mean_log_coeff[:self._num_cov_bases])))
      session.run(tf.assign(self._is_coupled, False))
      logging.info('Bases are now decoupled.')
    else:
      raise ValueError('Bases have already been decoupled.')

  def add_bases(self, session, num_bases_to_add, bases_to_sample=None):
    """Adds more bases.

    Bases are incrementally added. Before reaching the max number, the same
    bases are added to both mean and covariance. But they can be optimized
    separately based on the coupling condition.

    Args:
      session: tf session.
      num_bases_to_add: int, number of bases to add to mean and covariance.
      bases_to_sample: rank-2 tensor, to sample from for initialization.
    """

    num_mean_bases_old = session.run(self._num_mean_bases)
    num_cov_bases_old = session.run(self._num_cov_bases)
    is_coupled = session.run(self._is_coupled)
    num_mean_bases_new = min(num_mean_bases_old + num_bases_to_add,
                             self._mean_bases.shape.as_list()[0])
    num_cov_bases_new = min(num_cov_bases_old + num_bases_to_add,
                            self._cov_bases.shape.as_list()[0])

    # Decouple bases if mean and covariance will have different number of bases,
    # due to their different max number.
    if (num_mean_bases_new != num_cov_bases_new) and is_coupled:
      self.decouple_bases(session)
      is_coupled = session.run(self._is_coupled)

    # Sample bases to initialize the newly added bases.
    bases_init = None
    if bases_to_sample is not None:
      bases_init = self._build_bases_sampling(bases_to_sample, num_bases_to_add)

    if num_mean_bases_new > num_mean_bases_old:
      logging.info('Updating num_mean_bases to %d.', num_mean_bases_new)
      self._add_bases_worker(session, num_mean_bases_new, num_mean_bases_old,
                             self._num_mean_bases, self._mean_bases,
                             self._mean_stats, bases_init)
    # Update the number of bases for covariance even it's coupled
    # with the mean, mainly due to stats which is not shared.
    if is_coupled:
      session.run(
          tf.assign(self._num_cov_bases, session.run(self._num_mean_bases)))
    if num_cov_bases_new > num_cov_bases_old and (not is_coupled):
      logging.info('Updating num_cov_bases to %d.', num_cov_bases_new)
      self._add_bases_worker(session, num_cov_bases_new, num_cov_bases_old,
                             self._num_cov_bases, self._cov_bases,
                             self._cov_stats, bases_init)

  def _add_bases_worker(self, session, num_bases_new, num_bases_old, num_bases,
                        bases, stats, bases_init):
    """Adds bases for mean or covariance.

    Args:
      session: tf session.
      num_bases_new: int, new number of bases.
      num_bases_old: int, old number of bases.
      num_bases: rank-0 tensor, number of bases.
      bases: rank-2 tensor, private.
      stats: rank-2 tensor, private.
      bases_init: rank-2 tensor or array, to initialize the new bases.
    """
    # Compensate stats for normalization.
    # New value of normalizer is read because it depends on num_bases, which
    # is just updated. So as the public stats. In order to keep the public
    # stats the same, we compensate for the change in normalizer by scaling
    # private stats.

    # Compute new value.
    stats_normalizer_old = session.run(
        self._build_stats_normalizer(num_bases_old, stats))
    stats_normalizer_new = session.run(
        self._build_stats_normalizer(num_bases_new, stats))
    stats_new = session.run(stats) * stats_normalizer_new / stats_normalizer_old

    # Assign.
    session.run(tf.assign(stats, stats_new))
    session.run(tf.assign(num_bases, num_bases_new))

    # Initialize newly added bases.
    if bases_init is not None:
      session.run(
          tf.scatter_update(bases, tf.range(num_bases_old, num_bases_new),
                            bases_init[:num_bases_new - num_bases_old]))

  # In order to satisfy tf.cond arguments types, and make code succinct.
  # pylint: disable=unnecessary-lambda
  # pylint: disable=g-long-lambda
  def build_conditional_bases_adding(self,
                                     global_step,
                                     bases_adding_freq,
                                     num_bases_to_add,
                                     bases_to_sample=None):
    """Build bases adding in tf graph that happens at certain global steps.

    Update public variables that subsume the logic of adding bases at certain
    global steps.

    Args:
      global_step: rank-0 tensor, global step in optimization.
      bases_adding_freq: int, adding bases per bases_adding_freq optimization.
      steps.
      num_bases_to_add: int, number of bases to add before reaching the limit.
      bases_to_sample: rank-2 tensor or array.
    """
    bases_adding = tf.cond(
        tf.equal(tf.mod(global_step, bases_adding_freq), 0),
        lambda: self._build_bases_adding(num_bases_to_add, bases_to_sample),
        lambda: tf.group())
    # Public variables which are non tf variables have to be dependent on the
    # bases adding op, since tf variables that they depends on have been
    # assigned new values. Through this way, the assignment will be reflected in
    # a single session run.
    with tf.control_dependencies([bases_adding]):
      self._build_public_variables()

  def _build_bases_adding(self, num_bases_to_add, bases_to_sample=None):
    """Build operation that adds bases.

    Args:
      num_bases_to_add: number of bases to add to mean and covariance.
      bases_to_sample: rank-2 tensor or array, to sample from for
        initialization.
    Returns:
      tf operation.
    """
    num_mean_bases_new = tf.minimum(self._num_mean_bases + num_bases_to_add,
                                    self._mean_bases.shape[0])
    num_cov_bases_new = tf.minimum(self._num_cov_bases + num_bases_to_add,
                                   self._cov_bases.shape[0])
    # Decouple bases if mean and covariance will have different number of bases,
    # due to their different max number.
    bases_decoupling = tf.cond(
        tf.logical_and(
            tf.not_equal(num_mean_bases_new, num_cov_bases_new),
            self._is_coupled), self._build_bases_decoupling, lambda: tf.group())

    # Decoupling is done first.
    with tf.control_dependencies([bases_decoupling]):
      # Sample bases to initialize the newly added bases.
      bases_init = None
      if bases_to_sample is not None:
        bases_init = self._build_bases_sampling(bases_to_sample,
                                                num_bases_to_add)
      mean_bases_adding = tf.cond(
          tf.greater(num_mean_bases_new, self._num_mean_bases),
          lambda: self._build_bases_adding_worker(self._num_mean_bases,
                                                  num_mean_bases_new,
                                                  self._mean_bases,
                                                  self._mean_stats,
                                                  bases_init),
          lambda: tf.group())
      cov_bases_adding = tf.cond(
          tf.logical_and(
              tf.greater(num_cov_bases_new, self._num_cov_bases),
              tf.logical_not(self._is_coupled)),
          lambda: self._build_bases_adding_worker(self._num_cov_bases,
                                                  num_cov_bases_new,
                                                  self._cov_bases,
                                                  self._cov_stats,
                                                  bases_init),
          lambda: tf.group())
      num_cov_bases = tf.cond(
          self._is_coupled,
          lambda: tf.group(tf.assign(self._num_cov_bases, num_cov_bases_new)),
          lambda: tf.group())
      mean_bases_adding = tf.Print(mean_bases_adding, [self._num_mean_bases],
                                   'number of mean bases: ')
      num_cov_bases = tf.Print(num_cov_bases, [self._num_cov_bases],
                               'number of cov bases: ')

    return tf.group(mean_bases_adding, cov_bases_adding, num_cov_bases)

  # pylint: enable=unnecessary-lambda
  # pylint: enable=g-long-lambda

  def _build_bases_sampling(self, bases_to_sample, num_bases_to_sample):
    """Samples bases.

    Assuming the first k bases are the same with different num_bases_to_sample.

    Args:
      bases_to_sample: rank-2 tensor or array to sample bases from.
      num_bases_to_sample: number of bases to sample.
    Returns:
      rank-2 tensor.
    """
    bases_to_sample = tf.convert_to_tensor(
        bases_to_sample, dtype=utils.tf_float)
    idx = tf.random_uniform(
        shape=[num_bases_to_sample],
        maxval=tf.cast(tf.shape(bases_to_sample)[0], utils.tf_int),
        dtype=utils.tf_int)
    return tf.gather(bases_to_sample, idx)

  def _build_bases_decoupling(self):
    """Build operation that decouples bases."""
    bases_assign = tf.scatter_update(self._cov_bases,
                                     tf.range(self._num_cov_bases),
                                     self._mean_bases[:self._num_cov_bases])
    log_coeff_assign = tf.scatter_update(
        self._cov_log_coeff, tf.range(self._num_cov_bases),
        self._mean_log_coeff[:self._num_cov_bases])
    is_coupled_assign = tf.assign(self._is_coupled, False)
    bases_assign = tf.Print(bases_assign, [bases_assign], 'bases are decoupled')
    return tf.group(bases_assign, log_coeff_assign, is_coupled_assign)

  def _build_bases_adding_worker(self, num_bases, num_bases_new, bases, stats,
                                 bases_init):
    """Build operation that adds bases for mean or covariance.

    Args:
      num_bases: rank-0 tensor.
      num_bases_new: rank-0 tensor, new number of bases.
      bases: rank-2 tensor, private.
      stats: rank-0 tensor, private.
      bases_init: rank-2 tensor.
    Returns:
      tf operation.
    """
    stats_normalizer_new = self._build_stats_normalizer(num_bases_new, stats)
    stats_normalizer = self._build_stats_normalizer(num_bases, stats)
    stats_assign = tf.assign(stats,
                             stats * stats_normalizer_new / stats_normalizer)
    bases_assign = tf.group()
    if bases_init is not None:
      bases_assign = tf.scatter_update(bases, tf.range(
          num_bases, num_bases_new), bases_init[:num_bases_new - num_bases])
    # Now it's time to update the number of bases.
    with tf.control_dependencies([bases_assign, stats_assign]):
      num_bases = tf.assign(num_bases, num_bases_new)
    return tf.group(num_bases)

  def build_summaries(self):
    tf.summary.scalar(self._num_mean_bases.name, self._num_mean_bases)
    tf.summary.scalar(self._num_cov_bases.name, self._num_cov_bases)
