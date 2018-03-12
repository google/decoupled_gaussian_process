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
"""A tensorflow implementation of the decoupled Gaussian process model.

Decoupled Gaussian process decouples the representation of mean and covariance
in reproducing kernel Hilbert space. It can be optimized through stochastic
gradient-based algorithms, e.g. adam.

Paper:
Cheng, Ching-An, and Byron Boots. "Variational Inference for Gaussian Process
Models with Linear Complexity." Advances in Neural Information Processing
Systems. 2017.
http://papers.nips.cc/paper/7103-variational-inference-for-gaussian-process-models-with-linear-complexity
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
from decoupled_gaussian_process import bases_manager
from decoupled_gaussian_process import utils
import enum
import tensorflow as tf

CommonTerms = collections.namedtuple(
    'CommonTerms', ['k_cov_bases', 'inverse_in_cov', 'logdet_eye_quadratic'])


class KernelType(enum.Enum):
  """Supported types of kernel.

    auto relevance determination squared exponential kernel (ARD_SE), and
    auto relevance determination general squared exponential kernel (ARD_GSE).

    Details about ARD_SE kernel can be found in the Chapter 5.1 of the Gaussian
    Process for Machine Learning book:
    http://www.gaussianprocess.org/gpml/chapters/RW.pdf.
    For more information on the ARD_GSE kernel, please see the Appendix of the
    decoupled Gaussian process paper.
  """
  ARD_SE = 1
  ARD_GSE = 2


class DecoupledGaussianProcess(object):
  """Implementation for decoupled Gaussian process model."""

  def __init__(self,
               num_data_points,
               dim_bases,
               num_mean_bases,
               num_cov_bases=None,
               num_mean_bases_init=None,
               num_cov_bases_init=None,
               length_scale=None,
               signal_stddev=1.0,
               likelihood_variance=1.0,
               kernel_type=KernelType.ARD_SE,
               bases_to_sample_from=None,
               minibatch_size_mean_bases=None,
               seed=0,
               with_normalization=True,
               with_preconditioning=True):
    """Initializes the properties.

    The mode is applicable to general kernel and likelihood function. The
    current implementation supports two squared exponential kernels and Gaussian
    likelihood.

    Args:
      num_data_points: number of data will be avaiable, data can be 2d array or
        rank-2 tensor.
      dim_bases: the dimension of bases.
      num_mean_bases: number of bases for mean.
      num_cov_bases: number of bases for covariance.
      num_mean_bases_init: number of bases for mean at the beginning.
      num_cov_bases_init: number of bases for covariance at the beginning.
      length_scale: initial value for kernel length scale.
      signal_stddev: initial value for the standard deviation of kernel
        variance.
      likelihood_variance: initial value for the variance of Gaussian
        likelihood.
      kernel_type: KernelType, type of the kernel.
      bases_to_sample_from: rank-2 tensor or array, to initialize all the bases.
      minibatch_size_mean_bases: number of mean bases to use in part of the
        quadratic term with k_mean_bases.
      seed: scalar, random seed.
      with_normalization: bool, whether stats will be normalized.
      with_preconditioning: bool, whether stats will be preconditioned.
    """

    # Randomness.
    tf.set_random_seed(seed)

    # Parameters.
    if num_cov_bases is None:
      num_cov_bases = num_mean_bases
    if num_mean_bases_init is None:
      num_mean_bases_init = num_mean_bases
    if num_cov_bases_init is None:
      num_cov_bases_init = num_cov_bases
    self.with_preconditioning = with_preconditioning
    self.minibatch_size_mean_bases = minibatch_size_mean_bases
    self.num_data_points = num_data_points
    self.kernel_type = kernel_type

    # Hyperparameters.
    if length_scale is None:
      length_scale = tf.ones([dim_bases], dtype=utils.tf_float)
    if signal_stddev is None:
      signal_stddev = 1.0
    if likelihood_variance is None:
      likelihood_variance = 1.0
    with tf.variable_scope('hyperparameters'):
      self._log_length_scale = tf.get_variable(
          'log_length_scale',
          initializer=tf.cast(tf.log(length_scale), utils.tf_float),
          dtype=utils.tf_float)
      self._log_signal_stddev = tf.get_variable(
          'log_signal_stddev',
          initializer=tf.cast(tf.log(signal_stddev), utils.tf_float),
          dtype=utils.tf_float)
      self._log_likelihood_variance = tf.get_variable(
          'likelihood_variance',
          initializer=tf.cast(tf.log(likelihood_variance), utils.tf_float),
          dtype=utils.tf_float)

    # All dependencies on hyperparameters should go through them.
    self.log_length_scale = tf.identity(self._log_length_scale)
    self.log_signal_stddev = tf.identity(self._log_signal_stddev)
    self.log_likelihood_variance = tf.identity(self._log_likelihood_variance)

    # Bases manager for incrementally adding bases.
    with tf.variable_scope('variational_parameters'):
      self.bases = bases_manager.BasesMananger(
          dim_bases, num_mean_bases, num_cov_bases, num_mean_bases_init,
          num_cov_bases_init, bases_to_sample_from, with_normalization)

    # Add Jacobi preconditioning.
    if self.with_preconditioning:
      self.add_jacobi_preconditioner()

    # Build common terms.
    self.common_terms = self._build_common_terms()

  def add_jacobi_preconditioner(self):
    """Add normalization with Jacobi preconditioner."""
    self.bases.mean_stats /= tf.reshape(
        self.build_kernel_matrix(
            self.bases.mean_bases,
            coeff_a=self.bases.mean_coeff,
            diagonal_only=True), [-1, 1])
    self.bases.cov_stats /= tf.reshape(
        self.build_kernel_matrix(
            self.bases.cov_bases,
            coeff_a=self.bases.cov_coeff,
            diagonal_only=True), [-1, 1])

  def build_prediction(self, x, back_prop=True, diagonal_only=True):
    """Builds tensor for predicting mean and covariance.

    Args:
      x: rank-2 tensor. Inputs for prediction.
      back_prop: bool, indicates whether back-propagation is desired in the
        while_loop for building mean.
      diagonal_only: bool, indicates whether only diagonal terms in the cov
        are needed.
    Returns:
      rank-2 tf tensors for predictive mean and covariance, regardless of the
      diagonal_only flag.
    """
    # Predictive mean.
    # Due to the potentially large number of bases for mean which can also
    # increase gradually, we use while loop to loop over mean bases with batch
    # size `minibatch_size_mean_bases`.
    if self.minibatch_size_mean_bases is None:
      k_mean_bases = self.build_kernel_matrix(
          x, self.bases.mean_bases, coeff_b=self.bases.mean_coeff)
      mean = tf.matmul(k_mean_bases, self.bases.mean_stats)
    else:
      batch_size = self.minibatch_size_mean_bases
      # Can not access tf variable num_mean_bases, since it may not be assigned
      # new value before tf.shape is computed. However, self.bases.mean_bases
      # is tf op that depends on the assignment.
      total_size = tf.shape(self.bases.mean_bases)[0]
      n_batch = tf.cast(tf.ceil(total_size / batch_size), utils.tf_int)
      initial = (0, tf.TensorArray(utils.tf_float, size=n_batch))

      def body(i, mean):
        start, end = i * batch_size, tf.minimum((i + 1) * batch_size,
                                                total_size)
        mean_bases_sub = self.bases.mean_bases[start:end]
        mean_coeff_sub = self.bases.mean_coeff[start:end]
        a_sub = self.bases.mean_stats[start:end]
        k_mean_bases_sub = self.build_kernel_matrix(
            x, mean_bases_sub, coeff_b=mean_coeff_sub)
        return (i + 1, mean.write(i, tf.matmul(k_mean_bases_sub, a_sub)))

      cond = lambda i, _: i < n_batch
      _, mean = tf.while_loop(cond, body, initial, back_prop=back_prop)
      mean = tf.reduce_sum(mean.stack(), axis=0)

    # Predictive cov.
    k_cov_bases = self.build_kernel_matrix(
        x, self.bases.cov_bases, coeff_b=self.bases.cov_coeff)
    k_inverse_in_cov = tf.matmul(k_cov_bases, self.common_terms.inverse_in_cov)
    if diagonal_only:
      k_diagonal = self.build_kernel_matrix(x, diagonal_only=True)
      cov = k_diagonal - tf.reduce_sum(k_inverse_in_cov * k_cov_bases, axis=1)
      cov = tf.reshape(cov, (-1, 1))
    else:
      cov = self.build_kernel_matrix(x) - tf.matmul(
          k_inverse_in_cov, k_cov_bases, transpose_b=True)
    return mean, cov

  def build_expected_log_likelihood(self, mean, variance, y):
    """Builds tensor for the expected likelihood part of objetive.

    el = num_data_points / n * sum_i E_{q(f(x_i))} log p(y_i | f(x_i)).
    Here Gaussian likelihood is implemented: y_i = f(x_i) + epsilon_i, where
    epsilon_i is zero mean Gaussian.

    Args:
      mean: rank-2 column tensor, mean of f(x_i).
      variance: rank-2 column tensor, variance of f(x_i).
      y: rank-2 column tensor, targets or observation of f(x_i).
    Returns:
      rank-0 tensor.

    """
    scale = tf.cast(self.num_data_points / tf.shape(mean)[0], utils.tf_float)
    ell = -0.5 * tf.cast(tf.log(2.0 * math.pi), utils.tf_float)
    ell -= 0.5 * self.log_likelihood_variance
    ell -= (0.5 * (
        tf.square(y - mean) + variance) / tf.exp(self.log_likelihood_variance))
    ell = tf.reduce_sum(ell) * scale
    return ell

  def build_kl_divergence(self):
    """Builds tensor for the Kullback-Leibler divergence part of objective.

    Returns:
      rank-0 tensor.
    """
    kl = 0.5 * self._build_kernel_quadratic_subsampling()
    kl += 0.5 * self.common_terms.logdet_eye_quadratic
    kl -= 0.5 * tf.reduce_sum(
        tf.multiply(self.common_terms.k_cov_bases,
                    self.common_terms.inverse_in_cov))
    return kl

  def _build_kernel_quadratic_subsampling(self):
    """Builds mean_stats' K(mean_bases, mean_bases) mean_stats with sampling.

    Based on the size of a, generate offset to sample a certain number of items.
    Now just sample a CONTINGENT subset, instead of usig stride or scattered
    samples.

    Returns:
      rank-0 tensor.
    """
    # TODO(xinyanyan) Try other sampling strategy.
    mean_stats = self.bases.mean_stats
    mean_bases = self.bases.mean_bases
    mean_coeff = self.bases.mean_coeff

    if self.minibatch_size_mean_bases is None:
      mean_stats_sub = mean_stats
      mean_bases_sub = mean_bases
      mean_coeff_sub = mean_coeff
      scale = 1.0
    else:
      total_size = tf.cast(tf.shape(mean_stats)[0], utils.tf_int)
      batch_size = self.minibatch_size_mean_bases
      # Make sure that batch size is smaller than total size.
      batch_size = tf.minimum(batch_size, total_size)
      offset = tf.random_uniform(
          shape=[], minval=0, maxval=total_size, dtype=utils.tf_int)
      # TODO(xinyanyan) tf.gather gives memory warnings. However, in the tf.cond
      # way (two cases: whether offset + batch_size is larger than total_size or
      # not, sometimes cholesky will fail after tens of thousands of iterations.
      idx = tf.mod(tf.range(offset, offset + batch_size), total_size)
      mean_stats_sub = tf.gather(mean_stats, idx)
      mean_bases_sub = tf.gather(mean_bases, idx)
      mean_coeff_sub = tf.gather(mean_coeff, idx)
      scale = total_size / batch_size

    k_mean_bases_sub = self.build_kernel_matrix(
        mean_bases, mean_bases_sub, coeff_a=mean_coeff, coeff_b=mean_coeff_sub)
    quadratic_form = tf.matmul(
        mean_stats,
        tf.matmul(k_mean_bases_sub, mean_stats_sub),
        transpose_a=True)
    quadratic_form = tf.reshape(quadratic_form, [])  # reshape to a scalar
    quadratic_form *= tf.cast(scale, utils.tf_float)  # rescale

    return quadratic_form

  def _build_common_terms(self):
    """Builds tensors for common terms.

    Returns:
      `CommonTerms` object consisting of tensors.

    eye_quadratic = eye + cov_stats' k cov_stats.
    chol_factor = chol(eye_quadratic, lower=True).
    chol_solve_sol = chol_factor^{-1} cov_stats'
    inverse_in_cov = chol_solve_sol' chol_solve_sol.
    """
    k_cov_bases = self.build_kernel_matrix(
        self.bases.cov_bases, coeff_a=self.bases.cov_coeff)
    eye_quadratic = (
        tf.eye(tf.shape(self.bases.cov_stats)[0], dtype=utils.tf_float) +
        tf.matmul(
            self.bases.cov_stats,
            tf.matmul(k_cov_bases, self.bases.cov_stats),
            transpose_a=True))
    chol_factor = tf.cholesky(eye_quadratic)
    chol_solve_sol = tf.matrix_triangular_solve(
        chol_factor, tf.transpose(self.bases.cov_stats), lower=True)
    inverse_in_cov = tf.matmul(chol_solve_sol, chol_solve_sol, transpose_a=True)
    logdet_eye_quadratic = 2.0 * tf.reduce_sum(
        tf.log(tf.diag_part(chol_factor)))

    return CommonTerms(
        k_cov_bases=k_cov_bases,
        inverse_in_cov=inverse_in_cov,
        logdet_eye_quadratic=logdet_eye_quadratic)

  def build_kernel_matrix(self,
                          xa,
                          xb=None,
                          coeff_a=None,
                          coeff_b=None,
                          diagonal_only=False):
    r"""Builds auto relevance determination general squared exponential kernel.

    k(x, x') = \rho^2 \prod_d (2 l_{x,d} l_{x',d} / (l_{x,d}^2 +
                              l_{x',d}^2))^{1/2}) *
               exp(-\sum_d (|x_d - x'_d|^2 / (l_{x,d}^2 + l_{x',d}^2))
    And l_x is the length_scale of x = length_scale * (coeff of x).
    For more information on the ARD_SE kernel, please see Chapter 5.1 of the
    Gaussian Process for Machine Learning book:
    http://www.gaussianprocess.org/gpml/chapters/RW.pdf.
    For more information on the ARD_GSE kernel, please see the Appendix of the
    decoupled Gaussian process paper.
    Args:
      xa: rank-2 tensor.
      xb: rank-2 tensor.
      coeff_a: rank-2 tensor length scale coefficient for xa.
      coeff_b: rank-2 tensor length scale coefficient for xb.
      diagonal_only: bool, indicates whether only diagonal terms are needed.
    Returns:
      rank-1 tensor if `diagonal_only`, rank-2 tensor otherwise.
    Raises:
      ValueError: error in arguments.
    """
    # Only diagonal terms are needed.
    if diagonal_only:
      if xb is not None:
        raise ValueError('xb is not accepted for the diagonal only case.')
      return (tf.exp(2 * self.log_signal_stddev) * tf.ones(
          [tf.shape(xa)[0]], dtype=utils.tf_float))

    # Full kernel matrix is needed.
    # Prepare.
    scaled_xa = xa / tf.exp(self.log_length_scale)
    if xb is None:
      scaled_xb = scaled_xa
    else:
      scaled_xb = xb / tf.exp(self.log_length_scale)

    if self.kernel_type is KernelType.ARD_SE:
      k = tf.exp(2.0 * self.log_signal_stddev -
                 0.5 * utils.build_squared_distance(scaled_xa, scaled_xb))

    elif self.kernel_type is KernelType.ARD_GSE:
      if coeff_a is None:
        coeff_a = tf.ones(tf.shape(xa), dtype=utils.tf_float)
      if xb is None:
        coeff_b = coeff_a
      elif coeff_b is None:
        coeff_b = tf.ones(tf.shape(xb), dtype=utils.tf_float)

      expanded_coeff_a = tf.expand_dims(coeff_a, 1)
      expanded_coeff_b = tf.expand_dims(coeff_b, 0)
      prod_coeff = tf.multiply(expanded_coeff_a, expanded_coeff_b)
      sum_squared_expanded_coeff = (
          tf.square(expanded_coeff_a) + tf.square(expanded_coeff_b))
      multiplier = tf.sqrt(
          tf.reduce_prod(
              (2.0 * prod_coeff / sum_squared_expanded_coeff), axis=2))
      nominator_in_exp = tf.squared_difference(
          tf.expand_dims(scaled_xa, 1), tf.expand_dims(scaled_xb, 0))
      denominator_in_exp = sum_squared_expanded_coeff
      k = multiplier * tf.exp(2.0 * self.log_signal_stddev - tf.reduce_sum(
          nominator_in_exp / denominator_in_exp, axis=2))

    else:
      raise ValueError('Unsupported kernel type: {}'.format(self.kernel_type))

    return k

  # TODO(xinyanyan) These can be made much easier if we can get the subgraph
  # that depends on some ops inside some subgraphs passed in as arguments.
  # Currently we are manually tracking the dependency, which is tedious and
  # error-prone.
  def build_conditional_hyperparameter_initialization_and_bases_adding(
      self,
      global_step,
      x,
      y,
      bases_adding_freq,
      num_bases_to_add,
      percentile=50.0,
      num_data_points_for_pairwise_distance=None,
      bases_to_sample_from=None):
    """Build hyperparameter initialization and bases adding in tf graph.

    Update public variables that subsume the logic of initializing
    hyperparameters based on default heuristics at global step 0 and adding
    bases at certain global steps.

    Args:
      global_step: rank-0 tensor, global step in optimization.
      x: rank-2 tensor, inputs or features.
      y: rank-2 tensor, column, targets or labels.
      bases_adding_freq: will add bases per bases_adding_freq optimization
        steps.
      num_bases_to_add: int, number of bases to add before reaching the limit.
      percentile: in interval [0, 100], the percentile of pairwise distances of
        separate dimensions to initialize length scale.
      num_data_points_for_pairwise_distance: number of data points to compute
        pairwise distances for kernel length scale. `None` to use all.
      bases_to_sample_from: rank-2 tensor or array, for bases adding.
    """
    self._build_conditional_hyperparameter_initialization(
        global_step, x, y, percentile, num_data_points_for_pairwise_distance)
    self.build_conditional_bases_adding(global_step, bases_adding_freq,
                                        num_bases_to_add, bases_to_sample_from)

  def build_conditional_bases_adding(self,
                                     global_step,
                                     bases_adding_freq,
                                     num_bases_to_add,
                                     bases_to_sample_from=None):
    """Build bases adding in tf graph that happens at certain global steps.

    Update public variables that subsume the logic of adding bases at certain
    global steps.

    Args:
      global_step: rank-0 tensor, global step in optimization.
      bases_adding_freq: int, adding bases per bases_adding_freq optimization
      steps.
      num_bases_to_add: int, number of bases to add before reaching the limit.
      bases_to_sample_from: rank-2 tensor or array.
    """
    self.bases.build_conditional_bases_adding(
        global_step, bases_adding_freq, num_bases_to_add, bases_to_sample_from)

    # Bases adding should be done first, so that the changes can be reflected in
    # variables that depends on them. Otherwise, the behavior is undeterminstic.
    # Add Jacobi preconditioner for normalization of stats.
    if self.with_preconditioning:
      self.add_jacobi_preconditioner()
    # Build common terms.
    self.common_terms = self._build_common_terms()

  # In order to satisfy tf.cond arguments types, and make code succinct.
  # pylint: disable=unnecessary-lambda
  # pylint: disable=g-long-lambda
  def _build_conditional_hyperparameter_initialization(
      self,
      global_step,
      x,
      y,
      percentile=50.0,
      num_data_points_for_pairwise_distance=None):
    """Build hyperparameter initialization that happens at global step 0.

    Args:
      global_step: rank-0 tensor, global step in optimization.
      x: rank-2 tensor, inputs or features.
      y: rank-2 tensor, column, targets or labels.
      percentile: in interval [0, 100], the percentile of pairwise distances of
        separate dimensions to initialize length scale.
      num_data_points_for_pairwise_distance: number of data points to compute
        pairwise distances for kernel length scale. `None` to use all.
    """
    conditional_hyperparameter_initialization = tf.cond(
        tf.equal(global_step, 0),
        lambda: self._build_hyperparameter_initialization(
            x, y, percentile, num_data_points_for_pairwise_distance),
        lambda: tf.group())
    with tf.control_dependencies([conditional_hyperparameter_initialization]):
      self.log_length_scale = tf.identity(self._log_length_scale)
      self.log_signal_stddev = tf.identity(self._log_signal_stddev)
      self.log_likelihood_variance = tf.identity(self._log_likelihood_variance)

  # pylint: enable=unnecessary-lambda
  # pylint: enable=g-long-lambda

  def _build_hyperparameter_initialization(
      self, x, y, percentile=50.0, num_data_points_for_pairwise_distance=None):
    """Build tf operation for hyperparameter initialization..

    It is made private since it HAS to be used with and called before
    build_variables_conditional_bases_adding(), since they update
    different public variables and the public variables updated by
    build_conditional_bases_adding() have to be updated for conditional
    hyperparameter initialization too.

    Args:
      x: rank-2 tensor, inputs or features.
      y: rank-2 tensor, column, targets or labels.
      percentile: in interval [0, 100], the percentile of pairwise distances of
        separate dimensions to initialize length scale.
      num_data_points_for_pairwise_distance: number of data points to compute
        pairwise distances for kernel length scale. `None` to use all.
    Returns:
      tf operation.
    """
    x = tf.convert_to_tensor(x, dtype=utils.tf_float)
    y = tf.convert_to_tensor(y, dtype=utils.tf_float)

    # Length scale.
    if num_data_points_for_pairwise_distance is None:
      idx = tf.range(tf.shape(x)[0])
    else:
      idx = tf.random_uniform(
          shape=[num_data_points_for_pairwise_distance],
          minval=0,
          maxval=tf.shape(x)[0],
          dtype=utils.tf_int)
    sampled_x = tf.gather(x, idx)
    initial = (0,
               tf.TensorArray(
                   utils.tf_float, size=sampled_x.shape.as_list()[1]))

    def body(i, length_scale):
      distances = tf.reshape(
          tf.abs(
              tf.expand_dims(sampled_x[:, i], 1) -
              tf.expand_dims(sampled_x[:, i], 0)), [-1])
      l = tf.contrib.distributions.percentile(
          distances, percentile, interpolation='nearest')
      return (i + 1, length_scale.write(i, l))

    cond = lambda i, _: i < sampled_x.shape.as_list()[1]
    _, length_scale = tf.while_loop(cond, body, initial, back_prop=False)
    length_scale = length_scale.stack()
    length_scale *= tf.sqrt(tf.cast(x.shape[1], utils.tf_float))
    log_length_scale_assign = tf.assign(self._log_length_scale,
                                        tf.log(length_scale))
    # Signal standard deviation.
    _, y_variance = tf.nn.moments(tf.reshape(y, [-1]), axes=[0])
    signal_stddev = tf.sqrt(y_variance)
    log_signal_stddev_assign = tf.assign(self._log_signal_stddev,
                                         tf.log(signal_stddev))

    # Variance in Gaussian likelihood.
    log_likelihood_variance_assign = tf.assign(self._log_likelihood_variance,
                                               tf.log(y_variance / 100.0))

    return tf.group(log_length_scale_assign, log_signal_stddev_assign,
                    log_likelihood_variance_assign)
