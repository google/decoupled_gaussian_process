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
r"""Example for using decoupled Gaussian process model.

A simple example of using the decoupled Gaussian process model. It supports
embedding the logic of adding bases online and initialization of hyperparameters
in the tensorflow graph.

Dataset in a numpy npz file needs to be provided, with fields: 'x_training',
'y_training', 'x_test', 'y_test', all are rank-2 numpy arrays.

A command example:
python3 decoupled_gaussian_process_example.py \
  --dataset_path testdata/emm_ground_truth.npz --y_index 8 \
  --kernel_type ARD_SE --learning_rate_with_decay True --num_mean_bases 1024 \
  --percentile 10 --have_logic_embedded_in_graph True
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import time
import decoupled_gaussian_process
from decoupled_gaussian_process import utils
from decoupled_gaussian_process.utils import numpy_data_interface
from decoupled_gaussian_process.utils import performance_logger
import numpy as np
import tensorflow as tf

tf.flags.DEFINE_string(
    'dataset_path',
    'third_party/py/decoupled_gaussian_process/testdata/walker.npz',
    'Path to the dataset.')
tf.flags.DEFINE_integer('y_index', 8,
                        'The number of column of y to regress on.')
tf.flags.DEFINE_string('log_path', None, 'Path of np log file to be saved.')
tf.flags.DEFINE_integer('num_mean_bases', 1024, 'Number of bases for mean.')
tf.flags.DEFINE_integer(
    'num_mean_bases_init', 0,
    'Number of bases for mean in the beginning of training.')
tf.flags.DEFINE_integer('num_cov_bases_init', 0,
                        ('Number of bases for covariance in the beginning of '
                         'training.'))
tf.flags.DEFINE_integer('num_cov_bases', 128, 'Number of bases for covariance.')
tf.flags.DEFINE_float('percentile', 10.0,
                      'Percentile for length_scale initialization.')
tf.flags.DEFINE_float('learning_rate', 0.01, 'Learning rate in training.')
tf.flags.DEFINE_boolean('learning_rate_with_decay', True,
                        'Learning rate decays in training.')
tf.flags.DEFINE_integer('minibatch_size_data', 1024,
                        'Size of minibatch for data.')
tf.flags.DEFINE_integer('minibatch_size_mean_bases', 128,
                        'Size of minibatch for mean in training.')
tf.flags.DEFINE_integer('max_step', 100000, 'Max number of optimization steps.')
tf.flags.DEFINE_integer('eval_freq', 50,
                        'Frequency of evaluation during training.')
tf.flags.DEFINE_integer('bases_adding_freq', 500, 'Frequency of adding bases.')
tf.flags.DEFINE_integer('num_bases_to_add', 128,
                        'Number of bases at each bases adding call.')
# Memory will likely to be exhausted if num_test_data_points is large when using
# 'ARD_GSE' kernel. It's better to fed in minibatch in that case. Here we just
# use a small number of test data points, e.g. 1024. `-1` for using all test
# data points.
tf.flags.DEFINE_integer('num_test_data_points', -1,
                        'Number of bases at each bases adding call.')
# 'ARD_SE' or 'ARD_GSE'
tf.flags.DEFINE_string('kernel_type', 'ARD_SE', 'Type of the kernel.')
tf.flags.DEFINE_boolean('have_logic_embedded_in_graph', True,
                        ('Have the logic of adding bases and hyperparameter '
                         'initialization embedded in the tensorflow graph.'))

FLAGS = tf.flags.FLAGS


def run(data,
        minibatch_size_data,
        minibatch_size_mean_bases,
        num_mean_bases,
        num_cov_bases=None,
        num_mean_bases_init=128,
        num_cov_bases_init=128,
        num_bases_to_add=128,
        bases_adding_freq=500,
        percentile=50,
        learning_rate=1e-2,
        learning_rate_with_decay=False,
        max_step=1e5,
        eval_freq=50,
        seed=0,
        kernel_type=decoupled_gaussian_process.KernelType.ARD_SE,
        have_logic_embedded_in_graph=False):
  """Use decoupled Gaussian process.

  Args:
    data: Dataset object defined in numpy_data_interface.py.
    minibatch_size_data: the size of minibatch for data.
    minibatch_size_mean_bases: number of mean bases to use in part of the
      quadratic term with k_mean_bases.
    num_mean_bases: number of bases for mean.
    num_cov_bases: number of bases for cov.
    num_mean_bases_init: number of bases for mean at the beginning.
    num_cov_bases_init: number of bases for covariance at the beginning.
    num_bases_to_add: number of bases to added in bases update.
    bases_adding_freq: adding bases per bases_adding_freq optimization steps.
    percentile: percentile for length_scale initialization.
    learning_rate: learning rate.
    learning_rate_with_decay: bool, indicates whether learning rate decays.
    max_step: the maximum number of steps in optimization.
    eval_freq: frequency of evaluation.
    seed: random seed.
    kernel_type: KernelType, type of the kernel.
    have_logic_embedded_in_graph: bool, have the logic of adding bases, and etc
      embedded in tf graph.
  Returns:
    Logger object.
  """
  data_interface = numpy_data_interface.NumpyDataInterface(seed)
  data_interface.prepare_data_for_minibatch(
      data,
      minibatch_size_data,
      is_sequential=True,
      with_whitening=True,
      seed=seed)
  bases_to_sample_from = (
      data_interface.sample_normalized_training_x(num_mean_bases))
  if num_cov_bases is None:
    num_cov_bases = num_mean_bases

  # Initialize hyperparameters.
  # length_scale is initialized based on the number of bases to be added
  # incrementally.
  x_placeholder, y_placeholder, feed_dict = data_interface.build_feed_dict()
  # If hyperparameter initialization with default heuristics is not embedded in
  # the tf graph, we compute them beforehand.
  if not have_logic_embedded_in_graph:
    x_for_hyperparameter_initialization, y_for_hyperparameter_initialization = (
        data_interface.get_next_normalized_training_batch(False))
    hyperparameters = utils.init_hyperparameters(
        x=x_for_hyperparameter_initialization,
        y=y_for_hyperparameter_initialization,
        percentile=percentile,
        num_data_points_for_pairwise_distance=num_bases_to_add,
        random_state=np.random.RandomState(seed))
    length_scale, signal_stddev, likelihood_variance = hyperparameters
  else:
    bases_to_sample_from = None
    length_scale, signal_stddev, likelihood_variance = None, None, None

  model = decoupled_gaussian_process.DecoupledGaussianProcess(
      data.training.x.shape[0], data.training.x.shape[1], num_mean_bases,
      num_cov_bases, num_mean_bases_init, num_cov_bases_init, length_scale,
      signal_stddev, likelihood_variance, kernel_type, bases_to_sample_from,
      minibatch_size_mean_bases, seed)

  # Evaluation.
  # Accuracy / normalized mean squared error (nMSE) to evaluate on the test
  # dataset.
  mean, _ = model.build_prediction(x_placeholder, back_prop=False)
  _, y_variance = tf.nn.moments(tf.reshape(y_placeholder, [-1]), axes=[0])
  mean_squared_error = tf.reduce_mean(tf.square(mean - y_placeholder))
  normalized_mean_squared_error = mean_squared_error / y_variance

  # Train step.
  global_step = tf.Variable(0, trainable=False)
  r1 = learning_rate if learning_rate_with_decay else 0.0
  learning_rate = utils.build_learning_rate_with_decay(learning_rate, r1,
                                                       global_step)
  if have_logic_embedded_in_graph:
    # Embed in the logic of hyperparameter initialization and bases adding in
    # the tf graph.
    model.build_conditional_hyperparameter_initialization_and_bases_adding(
        global_step, x_placeholder, y_placeholder, bases_adding_freq,
        num_bases_to_add, percentile, num_bases_to_add, x_placeholder)
  mean, variance = model.build_prediction(x_placeholder, back_prop=True)
  evidence_lower_bound = (
      -model.build_kl_divergence() +
      model.build_expected_log_likelihood(mean, variance, y_placeholder))
  objective = tf.negative(evidence_lower_bound)
  train_step = (
      tf.train.AdamOptimizer(learning_rate).minimize(
          objective, global_step=global_step))

  # Initialize variables in graph.
  session = tf.Session()
  session.run(tf.global_variables_initializer())

  # Train.
  start_time = time.time()
  logger = performance_logger.PerformanceLogger([
      'step', 'learning_rate', 'normalized_mean_squared_error',
      'evidence_lower_bound', 'time'
  ])
  try:
    step = 0
    while step < max_step:
      # Add bases.
      if (have_logic_embedded_in_graph is False and
          step % bases_adding_freq == 0):
        model.bases.add_bases(
            session,
            num_bases_to_add,
            data_interface.get_next_normalized_training_batch(
                increase_counter=False)[0])
      # Train step.
      _, elbo = session.run(
          [train_step, evidence_lower_bound],
          feed_dict=feed_dict(during_training=True))
      # Evaluation.
      if step % eval_freq == 0:
        nmse = session.run(
            normalized_mean_squared_error,
            feed_dict=feed_dict(during_training=False))

      # Log.
      if step % eval_freq == 0:
        logger.append(
            step=step,
            learning_rate=session.run(learning_rate),
            normalized_mean_squared_error=nmse,
            evidence_lower_bound=elbo,
            time=time.time() - start_time)
        logger.print_appended()
      step += 1
  except KeyboardInterrupt:
    print('Caught KeyboardInterrupt, end training.')

  # Final evaluation.
  nmse, elbo = session.run(
      [normalized_mean_squared_error, evidence_lower_bound],
      feed_dict=feed_dict(during_training=False))
  logger.append(
      step=step,
      learning_rate=session.run(learning_rate),
      normalized_mean_squared_error=nmse,
      evidence_lower_bound=elbo,
      time=time.time() - start_time)
  logger.print_appended()

  return logger


def main(unused_args):
  del unused_args

  # Load data from file.
  seed = 11
  rng = np.random.RandomState(seed)
  np_data = np.load(FLAGS.dataset_path)

  # Random sample the test data points, because without the mechanism for
  # feeding small batches of test data, we can run out of memory when we use the
  # ARD_GSE kernel. It's better to feed in minibatches when testing.
  if FLAGS.num_test_data_points > 0:
    idx_test_data_points = rng.randint(0, len(np_data['y_test']),
                                       FLAGS.num_test_data_points)
  else:
    idx_test_data_points = np.arange(len(np_data['y_test']))
  x_training = np_data['x_training']
  y_training = np_data['y_training'][:, [FLAGS.y_index]]
  x_test = np_data['x_test'][idx_test_data_points]
  y_test = np_data['y_test'][idx_test_data_points][:, [FLAGS.y_index]]
  data = numpy_data_interface.Dataset(
      training=numpy_data_interface.DataPoints(x=x_training, y=y_training),
      test=numpy_data_interface.DataPoints(x=x_test, y=y_test))
  logging.basicConfig(level=logging.INFO)

  # Run.
  logger = run(data, FLAGS.minibatch_size_data, FLAGS.minibatch_size_mean_bases,
               FLAGS.num_mean_bases, FLAGS.num_cov_bases,
               FLAGS.num_mean_bases_init, FLAGS.num_cov_bases_init,
               FLAGS.num_bases_to_add, FLAGS.bases_adding_freq,
               FLAGS.percentile, FLAGS.learning_rate,
               FLAGS.learning_rate_with_decay, FLAGS.max_step, FLAGS.eval_freq,
               seed, decoupled_gaussian_process.KernelType[FLAGS.kernel_type],
               FLAGS.have_logic_embedded_in_graph)
  if FLAGS.log_path is not None:
    logging.info('saving log to: ' + FLAGS.log_path)
    logger.save(FLAGS.log_path)


if __name__ == '__main__':
  tf.flags.mark_flags_as_required(['dataset_path', 'y_index'])
  tf.app.run()
