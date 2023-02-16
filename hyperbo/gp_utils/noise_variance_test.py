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

"""Test noise_variance.py."""

import logging

from absl.testing import absltest
from absl.testing import parameterized
from flax import linen as nn
from hyperbo.basics import definitions as defs
from hyperbo.basics import params_utils
from hyperbo.gp_utils import basis_functions as bf
from hyperbo.gp_utils import noise_variance
import jax
import jax.numpy as jnp

GPParams = defs.GPParams
retrieve_params = params_utils.retrieve_params


class NoiseVarianceTest(parameterized.TestCase):
  """Tests for noise_variance.py."""

  def test_noise_variance_shape_and_value(self):
    """Test noise_variance vectors has expected shapes and values."""
    key = jax.random.PRNGKey(0)
    n = 15
    xdim = 5
    vx = jax.random.normal(key, (n, xdim))
    params = GPParams(model={'constant_noise_variance': 5.0})
    constant = noise_variance.constant(params, vx)
    self.assertEqual(constant.shape, (n, 1))
    # pylint: disable=expression-not-assigned
    for val in constant:
      self.assertAlmostEqual(val, params.model['constant_noise_variance'])

    params = GPParams(
        model={
            'linear_noise_variance': nn.Dense(1).init(key, vx)['params'],
        }
    )
    noise_variance_vals = noise_variance.linear(params, vx)
    self.assertEqual(noise_variance_vals.shape, (n, 1))
    (linear_noise_variance,) = retrieve_params(
        params, ['linear_noise_variance']
    )
    noise_variance_refs = (
        jnp.dot(vx, linear_noise_variance['kernel'])
        + linear_noise_variance['bias']
    )
    logging.info(
        msg=(
            f'linear noise_variance_vals={noise_variance_vals};'
            f' noise_variance_refs={noise_variance_refs}'
        )
    )
    self.assertAlmostEqual(
        jnp.sum(jnp.sqrt(noise_variance_vals - noise_variance_refs)), 0.0
    )

    params = GPParams(
        model={
            'linear_noise_variance': nn.Dense(1).init(key, jnp.empty((0, 8)))[
                'params'
            ],
        },
        config={
            'mlp_features': (8,),
        },
    )
    bf.init_mlp_with_shape(key, params, (0, xdim))
    noise_variance_vals = noise_variance.linear_mlp(params, vx)
    logging.info(msg=f'linear_mlp noise_variance_vals={noise_variance_vals}')
    self.assertEqual(noise_variance_vals.shape, (n, 1))

  @parameterized.parameters(
      noise_variance.softplus_warp, noise_variance.squared_warp
  )
  def test_noise_variance_warp(self, noise_warper):
    """Test warped noise_variance vectors has expected shapes and values."""
    warp_func = {'noise_variance': noise_warper}
    key = jax.random.PRNGKey(0)
    n = 15
    xdim = 5
    vx = jax.random.normal(key, (n, xdim))
    params = GPParams(model={'constant_noise_variance': 5.0})
    constant = noise_variance.constant(params, vx, warp_func=warp_func)
    self.assertEqual(constant.shape, (n, 1))
    # pylint: disable=expression-not-assigned
    for val in constant:
      self.assertAlmostEqual(
          val, noise_warper(params.model['constant_noise_variance'])
      )
      self.assertGreater(val, 0.0)

    params = GPParams(
        model={
            'linear_noise_variance': nn.Dense(1).init(key, vx)['params'],
        }
    )
    noise_variance_vals = noise_variance.linear(params, vx, warp_func=warp_func)
    self.assertEqual(noise_variance_vals.shape, (n, 1))
    (linear_noise_variance,) = retrieve_params(
        params, ['linear_noise_variance']
    )
    noise_variance_refs = noise_warper(
        jnp.dot(vx, linear_noise_variance['kernel'])
        + linear_noise_variance['bias']
    )
    logging.info(
        msg=(
            f'linear noise_variance_vals={noise_variance_vals};'
            f' noise_variance_refs={noise_variance_refs}'
        )
    )
    for val, ref in zip(noise_variance_vals, noise_variance_refs):
      self.assertAlmostEqual(val, ref)
      self.assertGreater(val, 0.0)

    params = GPParams(
        model={
            'linear_noise_variance': nn.Dense(1).init(key, jnp.empty((0, 8)))[
                'params'
            ],
        },
        config={
            'mlp_features': (8,),
        },
    )
    bf.init_mlp_with_shape(key, params, (0, xdim))
    noise_variance_vals = noise_variance.linear_mlp(params, vx, warp_func)
    logging.info(msg=f'linear_mlp noise_variance_vals={noise_variance_vals}')
    self.assertEqual(noise_variance_vals.shape, (n, 1))
    for val in noise_variance_vals:
      self.assertGreater(val, 0.0)


if __name__ == '__main__':
  absltest.main()
