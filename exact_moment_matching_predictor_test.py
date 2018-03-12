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
"""Tests for exact_moment_matching_predictor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import decoupled_gaussian_process
from decoupled_gaussian_process import exact_moment_matching_predictor
from decoupled_gaussian_process import utils
import numpy as np
import tensorflow as tf

# For some reason, it needs to be set that high.
# With float64, this can be much smaller, e.g. 1e-9. To change it to float64,
# just modified `np_float` and `tf_float` in utils.py.
RELATIVE_TOL = 1e-02
TEST_DATA_PATH = 'third_party/py/decoupled_gaussian_process/testdata'


def _get_test_data_path(file_name):
  """Returns absolute path for a file in testdata directory."""
  return os.path.join(TEST_DATA_PATH, file_name)


class ExactMomentMatchingPredictorTest(tf.test.TestCase):

  def setUp(self):
    self._session = tf.Session()

  def _get_model(self, num_bases, hyperparams):
    """Build Gaussian process model given hyperparameters."""
    log_length_scale = np.reshape(hyperparams[:-2], [-1])
    log_signal_stddev = hyperparams[-2]
    log_likelihood_variance = hyperparams[-1] * 2.0
    model = decoupled_gaussian_process.DecoupledGaussianProcess(
        10, len(log_length_scale), num_bases, num_bases, num_bases, num_bases,
        np.exp(log_length_scale), np.exp(log_signal_stddev),
        np.exp(log_likelihood_variance),
        decoupled_gaussian_process.KernelType.ARD_SE)
    return model

  def _initialize_model(self, model, x, beta, inverse_in_cov):
    """Assign variables in the model that will be needed by the predictor."""
    self._session.run(tf.assign(model.bases._mean_bases, x))
    self._session.run(tf.assign(model.bases._mean_stats, beta))
    model.bases._build_public_variables()
    model.common_terms = decoupled_gaussian_process.CommonTerms(
        k_cov_bases=None,
        inverse_in_cov=inverse_in_cov,
        logdet_eye_quadratic=None)

  def _do_the_test(self, models, ground_truth):
    predictor = (
        exact_moment_matching_predictor.ExactMomentMatchingPredictor(models))
    # Input distribution.
    mu = np.reshape(ground_truth['m'], [-1]).astype(utils.np_float)
    sigma = ground_truth['s'].astype(utils.np_float)

    # Ground truth.
    means_np = np.reshape(ground_truth['M'], [-1]).astype(utils.np_float)
    covs_np = ground_truth['S'].astype(utils.np_float)

    # Compute using tf.
    prediction_tf = predictor.build_prediction(mu, sigma)
    means_tf, covs_tf = self._session.run(prediction_tf)

    # Assert.
    self.assertAllClose(means_tf, means_np, rtol=RELATIVE_TOL)
    self.assertAllClose(covs_tf, covs_np, rtol=RELATIVE_TOL)

  def testSISO(self):
    """Sanity check for the single input single output case."""
    # ground truth from matlab
    path = _get_test_data_path('emm_ground_truth.npz')
    ground_truth = np.load(path, 'r')

    # Data for decoupled Gaussian process.
    x = np.reshape(ground_truth['input'], [-1, 1]).astype(utils.np_float)
    hyperparams = np.reshape(ground_truth['X'], [-1]).astype(utils.np_float)
    beta = np.reshape(ground_truth['beta'], [-1, 1]).astype(utils.np_float)
    inverse_in_cov = ground_truth['iK'].astype(utils.np_float)

    # Decoupled Gaussian process model.
    model = self._get_model(x.shape[0], hyperparams)
    models = [model]

    # Session.
    self._session.run(tf.global_variables_initializer())
    self._initialize_model(model, x, beta, inverse_in_cov)

    # Test.
    self._do_the_test(models, ground_truth)

  def testMIMO(self):
    """Test for the multiple input and multiple output case."""
    # ground truth from matlab
    path = _get_test_data_path('emm_ground_truth_2.npz')
    ground_truth = np.load(path, 'r')
    x = ground_truth['input']
    dim_y = ground_truth['target'].shape[1]

    # Decoupled Gaussian process model.
    models = []
    for idx_y in range(dim_y):
      hyperparams = ground_truth['X'][idx_y, :]
      with tf.variable_scope('model_{}'.format(idx_y)):
        models.append(self._get_model(x.shape[0], hyperparams))
    # Session.
    self._session.run(tf.global_variables_initializer())
    for idx_y in range(dim_y):
      beta = ground_truth['beta'][:, [idx_y]]
      inverse_in_cov = ground_truth['iK'][:, :, idx_y]
      self._initialize_model(models[idx_y], x, beta, inverse_in_cov)
    # Test.
    self._do_the_test(models, ground_truth)

  def tearDown(self):
    self._session.close()
    tf.reset_default_graph()


if __name__ == '__main__':
  tf.test.main()
