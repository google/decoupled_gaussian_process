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
"""A logger for logging and printing performance information."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import numpy as np


class PerformanceLogger(object):
  """A logger for saving performance to disk as numpy array and printing."""

  def __init__(self, keys):
    """Initialization.

    Args:
      keys: list, the keys for the entries to be logged.
    """
    self.keys = keys
    self.log = {}
    for k in keys:
      self.log[k] = []

  def append(self, **kwargs):
    """Appends an entry to log.

    Args:
      **kwargs: key-word arguments, represents an entry to be added.
    """
    for k, v in kwargs.items():
      self.log[k].append(v)

  def save(self, path):
    """Saves the log to file.

    Args:
      path: string, the path of the file.
    """
    log = copy.deepcopy(self.log)
    log = {k: np.array(v) for k, v in log.items()}
    np.savez(path, **log)

  def print_appended(self):
    """Prints the just appended log with proper format."""
    string = ''
    for k in self.keys:
      v = self.log[k][-1]
      if isinstance(v, int):
        string += k + ': {:6d}, '.format(v)
      elif isinstance(v, float) or isinstance(v, np.float32):
        string += k + ': {:9.3}, '.format(v)
      else:
        string += k + ': {}, '.format(v)
    logging.info(string)
