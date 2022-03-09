# coding=utf-8
# Copyright 2022 HyperBO Authors.
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

from jax.config import config
config.update('jax_debug_nans', True)
"""
import logging

from absl.testing import absltest
from absl.testing import parameterized
from hyperbo.basics import definitions as defs
from hyperbo.experiments import const
from hyperbo.gp_utils import gp
from hyperbo.gp_utils import kernel
from hyperbo.gp_utils import mean
import jax

GPParams = defs.GPParams
ACFUN = const.ACFUN


class AcFunTest(parameterized.TestCase):
  """Tests for acfun.py."""

  @parameterized.named_parameters(ACFUN.items())
  def test_acfun_shape(self, ac_func):
    """Test that values returned from functions in acfun has the right shape."""
    key = jax.random.PRNGKey(0)
    x_key, y_key, q_key = jax.random.split(key, 3)
    nx, nq = 20, 10
    dim = 5
    vx = jax.random.normal(x_key, (nx, dim))
    params = GPParams(
        model={
            'constant': 5.,
            'lengthscale': 1.,
            'signal_variance': 1.0,
            'noise_variance': 0.01,
        })
    mean_func = mean.constant
    cov_func = kernel.squared_exponential
    vy = gp.sample_from_gp(y_key, mean_func, cov_func, params, vx)
    x_queries = jax.random.normal(q_key, (nq, dim))

    model = gp.GP(
        dataset=[(vx, vy)],
        mean_func=mean_func,
        cov_func=cov_func,
        params=params)

    ac_eval = ac_func(model=model, sub_dataset_key=0, x_queries=x_queries)
    logging.info(msg=f'acfun = {ac_func.__name__}, eval = {ac_eval}')
    self.assertEqual(ac_eval.shape, (nq, 1))


if __name__ == '__main__':
  absltest.main()
