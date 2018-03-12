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
"""Model function for Decoupled Gaussian Process.

The model_fn is to be used with high-level tf APIs, e.g.,
tf.estimator.Estimator.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import decoupled_gaussian_process
from decoupled_gaussian_process import utils
import tensorflow as tf


def model_fn(features, labels, mode, params):
  """The same signature required by model_fn arg from tf.estimator.Estimator."""
  x = tf.concat(list(features.values()), 1)
  y = labels
  model = decoupled_gaussian_process.DecoupledGaussianProcess(
      params.num_training_data_points,
      x.shape.as_list()[1], params.num_mean_bases, params.num_cov_bases,
      params.num_mean_bases_init, params.num_cov_bases_init, None, None, None,
      decoupled_gaussian_process.KernelType[params.kernel_type], None,
      params.minibatch_size_mean_bases, params.seed)

  # Predict or Eval.
  if (mode == tf.estimator.ModeKeys.PREDICT or
      mode == tf.estimator.ModeKeys.EVAL):
    mean, variance = model.build_prediction(x, back_prop=False)
    # Predict.
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=mean)
    # Eval.
    if mode == tf.estimator.ModeKeys.EVAL:
      mean_squared_error = tf.reduce_mean(tf.square(mean - y))
      # Construct the `normalized mean squared error` metrics.
      mean_squared_error, mean_squared_error_update = (
          tf.metrics.mean_squared_error(y, mean))
      mean_y, mean_y_update = tf.metrics.mean(y)
      mean_squared_y, mean_squared_y_update = tf.metrics.mean(tf.square(y))
      update_ops = tf.group(mean_squared_error_update, mean_y_update,
                            mean_squared_y_update)
      metrics = {}
      metrics['mean_squared_error'] = (mean_squared_error, update_ops)
      y_variance = mean_squared_y - tf.square(mean_y)
      y_variance *= (
          params.num_test_data_points / (params.num_test_data_points - 1))
      metrics['normalized_mean_squared_error'] = (
          mean_squared_error / y_variance, update_ops)
      # TODO(xinyanyan) Note that this is an approximation due to the possible
      # sampling in build_kl_divergence(). Provide a non-sampling version
      # without backprop.
      evidence_lower_bound = (
          -model.build_kl_divergence() +
          model.build_expected_log_likelihood(mean, variance, y))
      metrics['evidence_lower_bound'] = tf.metrics.mean(
          tf.ones(
              (tf.shape(x)[0], 1), dtype=utils.tf_float) * evidence_lower_bound)
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=mean_squared_error, eval_metric_ops=metrics)

  # Train.
  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.train.get_or_create_global_step()
    r1 = params.learning_rate if params.learning_rate_with_decay else 0.0
    learning_rate = utils.build_learning_rate_with_decay(
        params.learning_rate, r1, global_step)
    model.build_conditional_hyperparameter_initialization_and_bases_adding(
        global_step, x, y, params.bases_adding_freq, params.num_bases_to_add,
        params.percentile, params.num_bases_to_add, x)
    # Add summaries.
    model.bases.build_summaries()
    tf.summary.scalar('learning_rate', learning_rate)
    mean, variance = model.build_prediction(x, back_prop=True)
    evidence_lower_bound = (
        -model.build_kl_divergence() +
        model.build_expected_log_likelihood(mean, variance, y))
    objective = tf.negative(evidence_lower_bound)
    tf.summary.scalar('objective', objective)
    train_step = (
        tf.train.AdamOptimizer(learning_rate).minimize(
            objective, global_step=global_step))
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=tf.reduce_mean(tf.square(mean - y)),
        train_op=train_step)
