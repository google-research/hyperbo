# coding=utf-8
# Copyright 2024 HyperBO Authors.
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

"""Test mean.py."""

import logging

from absl.testing import absltest
from flax import linen as nn
from hyperbo.basics import definitions as defs
from hyperbo.basics import params_utils
from hyperbo.gp_utils import basis_functions as bf
from hyperbo.gp_utils import mean
import jax
import jax.numpy as jnp

GPParams = defs.GPParams
retrieve_params = params_utils.retrieve_params


class MeanTest(absltest.TestCase):
  """Tests for mean.py."""

  def test_mean_shape_and_value(self):
    """Test mean vectors has expected shapes and values."""
    key = jax.random.PRNGKey(0)
    n = 15
    xdim = 5
    vx = jax.random.normal(key, (n, xdim))
    params = GPParams(model={'constant': 5.})
    constant = mean.constant(params, vx)
    self.assertEqual(constant.shape, (n, 1))
    # pylint: disable=expression-not-assigned
    [self.assertAlmostEqual(val, params.model['constant']) for val in constant]

    params = GPParams(model={
        'linear_mean': nn.Dense(1).init(key, vx)['params'],
    })
    mean_vals = mean.linear(params, vx)
    self.assertEqual(mean_vals.shape, (n, 1))
    linear_mean, = retrieve_params(params, ['linear_mean'])
    mean_refs = jnp.dot(vx, linear_mean['kernel']) + linear_mean['bias']
    logging.info(msg=f'linear mean_vals={mean_vals}; mean_refs={mean_refs}')
    self.assertAlmostEqual(jnp.sum(jnp.sqrt(mean_vals - mean_refs)), 0.)

    params = GPParams(
        model={
            'linear_mean': nn.Dense(1).init(key, jnp.empty((0, 8)))['params'],
        },
        config={
            'mlp_features': (8,),
        })
    bf.init_mlp_with_shape(key, params, (0, xdim))
    mean_vals = mean.linear_mlp(params, vx)
    logging.info(msg=f'linear_mlp mean_vals={mean_vals}')
    self.assertEqual(mean_vals.shape, (n, 1))


if __name__ == '__main__':
  absltest.main()
