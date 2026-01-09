# coding=utf-8
# Copyright 2025 HyperBO Authors.
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

"""Test acfun.py.

Use the following to debug nan.

from jax import config
config.update('jax_debug_nans', True)
"""
import logging

from absl.testing import absltest
from absl.testing import parameterized

from hyperbo.basics import definitions as defs
from hyperbo.bo_utils import bayesopt
from hyperbo.bo_utils import const
from hyperbo.bo_utils import data
from hyperbo.gp_utils import kernel
from hyperbo.gp_utils import mean
import jax
import jax.numpy as jnp
import numpy as np

GPParams = defs.GPParams
ACFUN = const.ACFUN


class BayesOptTest(parameterized.TestCase):
  """Tests for bayesopt.py."""

  @parameterized.named_parameters(ACFUN.items())
  def test_run_synthetic(self, ac_func):
    """Test bayesopt.run_synthetic."""
    key = jax.random.PRNGKey(0)
    params = GPParams(
        model={
            'constant': 5.,
            'lengthscale': 1.,
            'signal_variance': 1.0,
            'noise_variance': 0.01,
        },
        config={
            'method': 'adam',
            'learning_rate': 1e-5,
            'beta': 0.9,
            'max_training_step': 1
        })
    mean_func = mean.constant
    cov_func = kernel.squared_exponential

    dataset, sub_dataset_key, queried_sub_dataset = data.random(
        key=key,
        mean_func=mean_func,
        cov_func=cov_func,
        params=params,
        dim=5,
        n_observed=0,
        n_queries=30,
        n_func_historical=2,
        m_points_historical=10)
    self.assertLen(dataset, 3)
    for i in range(2):
      d = dataset[i]
      self.assertEqual(d.x.shape, (10, 5))
      self.assertEqual(d.y.shape, (10, 1))
      self.assertIsNone(d.aligned)
    logging.info(
        msg=f'dataset: {jax.tree.map(jnp.shape, dataset)}, '
        f'queried sub-dataset key: {sub_dataset_key}'
        f'queried sub-dataset: {jax.tree.map(jnp.shape, queried_sub_dataset)}')
    observations, queries, params = bayesopt.run_bayesopt(
        dataset=dataset,
        sub_dataset_key=sub_dataset_key,
        queried_sub_dataset=queried_sub_dataset,
        mean_func=mean_func,
        cov_func=cov_func,
        init_params=params,
        ac_func=ac_func,
        iters=3,
        init_random_key=key)

    logging.info(
        msg=f'observations: {observations}, best query:{max(queries[1])}')
    logging.info(msg=f'params:{params}')

    self.assertEqual(observations[0].shape, (3, 5))
    self.assertEqual(observations[1].shape, (3, 1))
    self.assertEqual(queries[0].shape, (5,))
    self.assertEqual(queries[1], max(queried_sub_dataset.y))


if __name__ == '__main__':
  absltest.main()
