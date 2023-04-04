# coding=utf-8
# Copyright 2023 HyperBO Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for utils."""
import logging

from absl.testing import absltest
from hyperbo.gp_utils import utils
import numpy as np


class UtilsTest(absltest.TestCase):

  def test_kl_multivariate_normal(self):
    np.random.seed(1)
    mu0 = np.random.uniform(-5, 5, (10,))
    mu1 = np.random.uniform(-5, 5, (10,))
    cov0 = np.random.uniform(-5, 5, (10, 100))
    cov0 = np.dot(cov0, cov0.T)
    cov1 = np.random.uniform(-5, 5, (10, 100))
    cov1 = np.dot(cov1, cov1.T)
    kl_01 = utils.kl_multivariate_normal(mu0, cov0, mu1, cov1, partial=False)
    logging.info(msg=f'kl_01 = {kl_01}')
    self.assertGreater(kl_01, 0)
    kl_00 = utils.kl_multivariate_normal(mu0, cov0, mu0, cov0, partial=False)
    logging.info(msg=f'kl_00 = {kl_00}')
    self.assertAlmostEqual(kl_00, 0, delta=1e-5)

  def test_degenerate_kl_multivariate_normal(self):
    np.random.seed(1)
    mu0 = np.random.uniform(-5, 5, (100,))
    mu1 = np.random.uniform(-5, 5, (100,))
    feat0 = np.random.uniform(-5, 5, (100, 5))
    cov0 = np.dot(feat0, feat0.T)
    cov1 = np.random.uniform(-5, 5, (100, 1000))
    cov1 = np.dot(cov1, cov1.T)
    kl_01 = utils.kl_multivariate_normal(
        mu0, cov0, mu1, cov1, partial=False)
    logging.info(msg=f'kl_01 = {kl_01}')
    self.assertGreater(kl_01, 0)
    self.assertLess(kl_01, np.inf)


if __name__ == '__main__':
  absltest.main()
