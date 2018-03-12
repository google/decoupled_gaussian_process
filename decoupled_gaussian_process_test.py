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
"""Tests for decoupled_gaussian_process."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import decoupled_gaussian_process
from decoupled_gaussian_process import utils
import numpy as np
from scipy import linalg as la
import tensorflow as tf


class TestDecoupledGaussianProcess(tf.test.TestCase):
  """Class for testing decoupled Gaussian process _model.

  Test by comparing tf solutions with np.
  """

  def setUp(self):
    # Control the randomness.
    self._seed = 0
    self._rng = np.random.RandomState(self._seed)

    # Parameters.
    self._dim_bases = 3
    self._signal_stddev = 0.8
    self._num_mean_bases = 33
    self._num_cov_bases = 23
    self._num_bases_init = 6
    self._num_bases_to_add = 4
    self._bases_adding_freq = 2
    self._likelihood_variance = 0.2
    self._num_data_points = 50
    self._num_steps = 30
    self._learning_rate = 1e-2

    # Data and initial values.
    self._data_range = [0.2, 1.5]
    self._x = self._uniform([self._num_data_points, self._dim_bases])
    self._y = self._uniform([self._num_data_points, 1])
    self._length_scale = self._uniform([self._dim_bases])
    self._signal_stddev = 0.88
    self._likelihood_variance = 0.23
    self._session = tf.Session()

    # Model.
    self._reset_model()
    self._session.run(tf.global_variables_initializer())

  def _reset_model(self, with_none_hyperparameters=False):
    self._session.close()
    tf.reset_default_graph()
    self._session = tf.Session()
    if with_none_hyperparameters:
      length_scale = None
      signal_stddev = None
      likelihood_variance = None
    else:
      length_scale = self._length_scale
      signal_stddev = self._signal_stddev
      likelihood_variance = self._likelihood_variance
    self._model = decoupled_gaussian_process.DecoupledGaussianProcess(
        self._num_data_points,
        self._dim_bases,
        self._num_mean_bases,
        self._num_cov_bases,
        self._num_bases_init,
        self._num_bases_init,
        length_scale,
        signal_stddev,
        likelihood_variance,
        bases_to_sample_from=self._x,
        seed=self._seed)

  def _uniform(self, shape):
    samples = self._rng.uniform(self._data_range[0], self._data_range[1], shape)
    return samples.astype(utils.np_float)

  def tearDown(self):
    self._session.close()
    tf.reset_default_graph()

  def test_build_kernel_matrix_ard_se(self):

    def compute_tf_solution(xa, xb, diagonal_only):
      kernel_matrix = self._model.build_kernel_matrix(
          xa, xb, diagonal_only=diagonal_only)
      return self._session.run(kernel_matrix)

    def compute_np_solution(xa, xb, diagonal_only):
      if xb is None:
        xb = xa
      kernel_matrix = np.zeros((xa.shape[0], xb.shape[0]))
      for i in range(xa.shape[0]):
        for j in range(xb.shape[0]):
          if i != j and diagonal_only:
            continue
          difference = xa[i] / self._length_scale - xb[j] / self._length_scale
          distance = np.dot(difference, difference)
          kernel_matrix[i, j] = self._signal_stddev**2 * np.exp(-0.5 * distance)
      if diagonal_only:
        return np.diag(kernel_matrix)
      else:
        return kernel_matrix

    def test(xa, xb, diagonal_only):
      kernel_matrix_tf = compute_tf_solution(xa, xb, diagonal_only)
      kernel_matrix_np = compute_np_solution(xa, xb, diagonal_only)
      self.assertAllClose(kernel_matrix_tf, kernel_matrix_np)

    # Prepare.
    self._model.kernel_type = decoupled_gaussian_process.KernelType.ARD_SE
    xa = self._uniform([self._num_data_points, self._dim_bases])
    xb = self._uniform([self._num_data_points, self._dim_bases])

    # Test different xa and xb.
    test(xa, xb, False)

    # Test xb being None.
    test(xa, None, False)

    # Test xb being None, and only computing diagonal terms.
    test(xa, None, True)

  def test_build_kernel_matrix_ard_gse(self):

    def compute_tf_solution(xa, xb, coeff_a, coeff_b):
      kernel_matrix = self._model.build_kernel_matrix(xa, xb, coeff_a, coeff_b)
      return self._session.run(kernel_matrix)

    def compute_np_solution(xa, xb, coeff_a, coeff_b):
      if coeff_a is None:
        coeff_a = np.ones(xa.shape, dtype=utils.np_float)
      if xb is None:
        xb, coeff_b = xa, coeff_a
      elif coeff_b is None:
        coeff_b = np.ones(xb.shape, dtype=utils.np_float)
      multiplier = np.ones((len(xa), len(xb)), dtype=utils.np_float)
      exp_term = np.zeros((len(xa), len(xb)), dtype=utils.np_float)
      for i in range(len(xa)):
        for j in range(len(xb)):
          for d in range(xa.shape[1]):
            prod = coeff_a[i, d] * coeff_b[j, d]
            sum_square = coeff_a[i, d]**2 + coeff_b[j, d]**2
            multiplier[i, j] *= np.sqrt(2.0 * prod / sum_square)
            square_diff = (xa[i, d] - xb[j, d])**2
            exp_term[i, j] += (
                square_diff / sum_square / (self._length_scale[d]**2))
      return self._signal_stddev**2 * multiplier * np.exp(-exp_term)

    def test(xa, xb, coeff_a, coeff_b):
      kernel_matrix_tf = compute_tf_solution(xa, xb, coeff_a, coeff_b)
      kernel_matrix_np = compute_np_solution(xa, xb, coeff_a, coeff_b)
      self.assertAllClose(kernel_matrix_tf, kernel_matrix_np)

    # Prepare.
    self._model.kernel_type = decoupled_gaussian_process.KernelType.ARD_GSE
    xa = self._uniform([self._num_data_points, self._dim_bases])
    coeff_a = self._uniform([self._num_data_points, self._dim_bases])
    xb = self._uniform([self._num_data_points, self._dim_bases])
    coeff_b = self._uniform([self._num_data_points, self._dim_bases])

    # Test.
    test(xa, xb, coeff_a, coeff_b)
    test(xa, None, coeff_a, None)
    test(xa, xb, None, coeff_b)
    test(xa, None, None, None)

  def _compute_common_terms_np(self):
    k_cov_bases = self._session.run(
        self._model.build_kernel_matrix(self._model.bases.cov_bases))
    cov_stats = self._session.run(self._model.bases.cov_stats)
    eye_quadratic = (
        np.eye(self._num_bases_init, dtype=utils.np_float) +
        np.dot(cov_stats.T, np.dot(k_cov_bases, cov_stats)))
    chol_factor = la.cholesky(eye_quadratic, lower=True)
    chol_solve_sol = la.solve_triangular(chol_factor, cov_stats.T, lower=True)
    inverse_in_cov = np.dot(chol_solve_sol.T, chol_solve_sol)
    logdet_eye_quadratic = np.sum(np.log(chol_factor.diagonal()))
    return decoupled_gaussian_process.CommonTerms(
        k_cov_bases=k_cov_bases,
        inverse_in_cov=inverse_in_cov,
        logdet_eye_quadratic=logdet_eye_quadratic)

  def test_build_common_terms(self):
    """Test common terms used in decoupled Gaussian process."""
    # Compute using tf.
    common_terms_tf = self._session.run(self._model.common_terms)

    # Compute using np.
    common_terms_np = self._compute_common_terms_np()

    # Assert.
    self.assertAllClose(common_terms_tf.inverse_in_cov,
                        common_terms_np.inverse_in_cov)
    self.assertAllClose(common_terms_tf.logdet_eye_quadratic,
                        common_terms_np.logdet_eye_quadratic)

  def test_build_kl_divergence(self):
    # Compute using tf.
    kl_divergence_tf = self._session.run(self._model.build_kl_divergence())

    # Compute using np.
    common_terms = self._compute_common_terms_np()
    mean_stats = self._session.run(self._model.bases.mean_stats)
    k_mean_bases = self._session.run(
        self._model.build_kernel_matrix(self._model.bases.mean_bases))
    kl_divergence_np = np.dot(mean_stats.T, np.dot(k_mean_bases,
                                                   mean_stats)).item()
    kl_divergence_np += 2 * common_terms.logdet_eye_quadratic
    kl_divergence_np -= np.sum(
        common_terms.k_cov_bases.T * common_terms.inverse_in_cov)
    kl_divergence_np *= 0.5

    # Assert.
    self.assertAllClose(kl_divergence_tf, kl_divergence_np)

  def test_build_prediction(self):
    # Compute using tf.
    mean_tf, cov_tf = self._session.run(
        self._model.build_prediction(
            self._x, back_prop=False, diagonal_only=False))
    _, cov_diagonal_tf = self._session.run(
        self._model.build_prediction(
            self._x, back_prop=False, diagonal_only=True))

    # Compute np solution.
    k_x_mean_bases = self._session.run(
        self._model.build_kernel_matrix(self._x, self._model.bases.mean_bases))
    mean_np = np.dot(k_x_mean_bases,
                     self._session.run(self._model.bases.mean_stats))
    common_terms = self._compute_common_terms_np()
    k_x_cov_bases = self._session.run(
        self._model.build_kernel_matrix(self._x, self._model.bases.cov_bases))
    k_x = self._session.run(self._model.build_kernel_matrix(self._x))
    cov_np = k_x - np.dot(k_x_cov_bases,
                          np.dot(common_terms.inverse_in_cov, k_x_cov_bases.T))

    # Assert.
    self.assertAllClose(mean_tf, mean_np)
    self.assertAllClose(cov_tf, cov_np)
    self.assertAllClose(cov_diagonal_tf, np.reshape(cov_np.diagonal(), [-1, 1]))

  def test_build_expected_log_likelihood(self):
    # Compute using tf.
    idx = list(range(2))
    mean, variance = self._session.run(
        self._model.build_prediction(self._x[idx]))
    ell_tf = self._session.run(
        self._model.build_expected_log_likelihood(mean, variance, self._y[idx]))

    # Compte using np.
    scale = self._x.shape[0] / len(idx)
    ell_np = -0.5 * np.log(2 * np.pi)
    ell_np -= 0.5 * np.log(self._likelihood_variance)
    ell_np -= (
        0.5 * ((self._y[idx] - mean)**2 + variance) / self._likelihood_variance)
    ell_np = np.sum(ell_np) * scale

    # Assert.
    self.assertAllClose(ell_tf, ell_np)

  def test_build_conditional_bases_adding(self):
    # In order to have close results between logic with and without embedded in
    # the tf graph.
    bases_to_sample_from = np.reshape(self._x[0], [1, -1])

    def compute_elbo(have_logic_embedded_in_graph):
      elbo_log = np.zeros(self._num_steps)
      global_step = tf.Variable(0, trainable=False)
      if have_logic_embedded_in_graph:
        self._model.build_conditional_bases_adding(
            global_step, self._bases_adding_freq, self._num_bases_to_add,
            bases_to_sample_from)
      mean, variance = self._model.build_prediction(self._x, back_prop=True)
      elbo = (
          -self._model.build_kl_divergence() +
          self._model.build_expected_log_likelihood(mean, variance, self._y))
      objective = tf.negative(elbo)
      train_step = (
          tf.train.AdamOptimizer(self._learning_rate).minimize(
              objective, global_step=global_step))
      self._session.run(tf.global_variables_initializer())
      for step in range(self._num_steps):
        if (not have_logic_embedded_in_graph and
            step % self._bases_adding_freq == 0):
          self._model.bases.add_bases(self._session, self._num_bases_to_add,
                                      bases_to_sample_from)
        _, elbo_log[step] = self._session.run([train_step, elbo])
      return elbo_log

    # Without logic embedded in graph.
    tf.set_random_seed(self._seed)
    self._reset_model()
    elbo_wo_embedded = compute_elbo(False)

    # With logic embedded in graph.
    tf.set_random_seed(self._seed)
    self._reset_model()
    elbo_w_embedded = compute_elbo(True)

    # Assert.
    self.assertAllClose(elbo_wo_embedded, elbo_w_embedded)

  def test_build_hyperparameter_initialization(self):

    # Without logic embedded in graph.
    hyperparameters_np = utils.init_hyperparameters(
        self._x, self._y, percentile=10.0)

    # With logic embedded in graph.
    self._reset_model(with_none_hyperparameters=True)
    global_step = tf.Variable(0, trainable=False)
    self._model._build_conditional_hyperparameter_initialization(
        global_step, self._x, self._y, percentile=10.0)
    length_scale = tf.exp(self._model.log_length_scale)
    signal_stddev = tf.exp(self._model.log_signal_stddev)
    likelihood_variance = tf.exp(self._model.log_likelihood_variance)
    self._session.run(tf.global_variables_initializer())
    hyperparameters_tf = self._session.run(
        [length_scale, signal_stddev, likelihood_variance])

    # Assert.
    for i in range(len(hyperparameters_np)):
      self.assertAllClose(hyperparameters_np[i], hyperparameters_tf[i])


if __name__ == '__main__':
  tf.test.main()
