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
"""Exact moment matching.

Exact moment matching for decoupled Gaussian process with squared exponential
kernel. Equations based on the paper:

Deisenroth, Marc Peter, Marco F. Huber, and Uwe D. Hanebeck. "Analytic
moment-based Gaussian process filtering." Proceedings of the 26th annual
international conference on machine learning. ACM, 2009.
https://spiral.imperial.ac.uk:8443/bitstream/10044/1/12195/4/icml2009_finalCorrected.pdf
(final corrected)

Matlab code: https://github.com/ICL-SML/gp-adf.
Especially, the file gpPt.m, transition propagation.
A side note: The equations in the matlab is quite different from the equations
in the paper. But actually they are the same, we need to use matrix identity and
the fact that Lambda are diagonal.

Small modifications are necessary for decoupled Gaussian process, e.g., taking
into account different inducing points / bases for mean and covariance, and
sampling for mean bases in order to achieve linear-time complexity:
"""

# TODO(xinyanyan) Add the computation for input-output covariance, which will
# be needed for filtering, and when delta x is predicted.
# TODO(xinyanyan) Add support for different bases for mean and covariance.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import decoupled_gaussian_process
from decoupled_gaussian_process import utils
import tensorflow as tf


class ExactMomentMatchingPredictor(object):
  """Class for predicting a sequence of belief states.

  The prediction is based on exact moment matching and decouple Gaussian process
  model.
  """

  def __init__(self, models):
    """Initialization.

    Args:
      models: list of DecoupledGaussianProcess.
    Raises:
      ValueError: invalid models are passed.
    """
    for model in models:
      if not isinstance(model,
                        decoupled_gaussian_process.DecoupledGaussianProcess):
        raise ValueError('Models include invalid model!')
    self.num_models = len(models)
    self.models = models

  def build_prediction(self, input_mean, input_cov):
    """Build exact moment matching predictions.

    Args:
      input_mean: rank-1 tensor.
      input_cov: rank-2 tensor.
    Returns:
      tensors for mean and covariance.
    """
    output_mean = [None for _ in range(self.num_models)]
    output_cov = [
        [None for _ in range(self.num_models)] for _ in range(self.num_models)
    ]
    k = [None for _ in range(self.num_models)]

    for i in range(self.num_models):
      quad_form = (
          input_cov + tf.diag(tf.exp(2.0 * self.models[i].log_length_scale)))
      coeff = (
          tf.exp(2.0 * self.models[i].log_signal_stddev) / tf.sqrt(
              tf.matrix_determinant(quad_form)) * tf.exp(
                  tf.reduce_sum(self.models[i].log_length_scale)))
      exp_distance = tf.exp(-0.5 * utils.build_squared_distance(
          self.models[i].bases.mean_bases,
          tf.reshape(input_mean, [1, -1]),
          matrix=quad_form,
          is_inverse=True))  # rank-2 column
      output_mean[i] = coeff * tf.reduce_sum(
          exp_distance * self.models[i].bases.mean_stats)
      scaled_x = self.models[i].bases.mean_bases - input_mean
      scaled_x /= tf.exp(self.models[i].log_length_scale)
      # rank-1, (num of bases,)
      k[i] = (2.0 * self.models[i].log_signal_stddev -
              tf.reduce_sum(scaled_x * scaled_x, axis=1) / 2.0)

    for i in range(self.num_models):
      scaled_xi = self.models[i].bases.mean_bases - input_mean
      scaled_xi /= tf.exp(2.0 * self.models[i].log_length_scale)
      for j in range(i + 1):
        quad_form = (
            input_cov * (tf.exp(-2.0 * self.models[i].log_length_scale) +
                         tf.exp(-2.0 * self.models[j].log_length_scale)) +
            tf.eye(tf.shape(input_cov)[0], dtype=utils.tf_float))
        coeff = 1.0 / tf.sqrt(tf.matrix_determinant(quad_form))
        scaled_xj = self.models[j].bases.mean_bases - input_mean
        scaled_xj /= tf.exp(2.0 * self.models[j].log_length_scale)
        exp_term = tf.reshape(k[i], [-1, 1]) + k[j]
        prod_over_sum = (
            tf.exp(2.0 * (self.models[i].log_length_scale +
                          self.models[j].log_length_scale)) /
            (tf.exp(2.0 * self.models[i].log_length_scale) +
             tf.exp(2.0 * self.models[j].log_length_scale)))
        exp_term += 0.5 * utils.build_squared_distance(
            scaled_xi, -scaled_xj, diagonal=prod_over_sum)
        exp_term -= 0.5 * utils.build_squared_distance(
            scaled_xi,
            -scaled_xj,
            matrix=quad_form,
            diagonal=prod_over_sum,
            is_inverse=True)
        exp_term = tf.exp(exp_term)
        multiplier_in_tr = tf.matmul(
            self.models[i].bases.mean_stats,
            self.models[j].bases.mean_stats,
            transpose_b=True)
        if i == j:
          multiplier_in_tr -= self.models[i].common_terms.inverse_in_cov
        # Get centered cov.
        output_cov[i][j] = (
            coeff * tf.reduce_sum(multiplier_in_tr * exp_term) -
            output_mean[i] * output_mean[j])
        output_cov[j][i] = output_cov[i][j]
      output_cov[i][i] += (
          tf.exp(2.0 * self.models[i].log_signal_stddev) +
          tf.exp(self.models[i].log_likelihood_variance))

    return output_mean, output_cov
