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
"""Tests for bases_manager."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from decoupled_gaussian_process import bases_manager
from decoupled_gaussian_process import utils
import numpy as np
import tensorflow as tf


class BasesManagerTest(tf.test.TestCase):

  def setUp(self):
    self._bases_to_sample_from = np.reshape(range(15), (5, 3)).astype(
        utils.np_float)
    self._num_mean_bases = 5
    self._num_cov_bases = 4
    self._num_mean_bases_init = 2
    self._num_cov_bases_init = 2
    self._num_bases_to_add = 3
    self._bases = bases_manager.BasesMananger(
        self._bases_to_sample_from.shape[1], self._num_mean_bases,
        self._num_cov_bases, self._num_mean_bases_init,
        self._num_cov_bases_init, self._bases_to_sample_from)
    self._session = tf.Session()
    self._session.run(tf.global_variables_initializer())

  def tearDown(self):
    tf.reset_default_graph()
    self._session.close()

  def test_decouple_bases(self):

    # Bases before decoupling.
    mean_bases_old = self._session.run(self._bases.mean_bases)
    cov_bases_old = self._session.run(self._bases.cov_bases)
    self.assertAllClose(mean_bases_old, cov_bases_old)

    # Decouple, and assign new values to cov_bases.
    self._bases.decouple_bases(self._session)
    self._session.run(
        tf.assign(
            self._bases._cov_bases,
            np.zeros(
                (self._num_cov_bases, self._bases_to_sample_from.shape[1]),
                dtype=utils.np_float)))

    # Bases after decoupling.
    mean_bases_new = self._session.run(self._bases.mean_bases)
    cov_bases_new = self._session.run(self._bases.cov_bases)
    self.assertAllClose(mean_bases_old, mean_bases_new)
    self.assertAllClose(cov_bases_new,
                        np.zeros(cov_bases_old.shape, dtype=utils.np_float))

  def test_add_bases(self):
    mean_stats_old = self._session.run(self._bases.mean_stats)
    cov_stats_old = self._session.run(self._bases.cov_stats)
    self._bases.add_bases(self._session, self._num_bases_to_add,
                          self._bases_to_sample_from)
    mean_stats_new = self._session.run(self._bases.mean_stats)
    cov_bases_new = self._session.run(self._bases.cov_stats)
    self.assertAllClose(
        self._session.run(self._bases._num_mean_bases),
        min(self._num_mean_bases,
            self._num_bases_to_add + self._num_mean_bases_init))

    self.assertAllClose(
        self._session.run(self._bases._num_cov_bases),
        min(self._num_cov_bases,
            self._num_bases_to_add + self._num_cov_bases_init))
    self.assertAllClose(mean_stats_old,
                        mean_stats_new[:self._num_mean_bases_init])
    self.assertAllClose(
        cov_stats_old,
        cov_bases_new[:self._num_cov_bases_init, :self._num_cov_bases_init])


if __name__ == '__main__':
  tf.test.main()
