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

"""Test kernel.py."""

import logging

from absl.testing import absltest
from absl.testing import parameterized
from hyperbo.basics import definitions as defs
from hyperbo.gp_utils import basis_functions as bf
from hyperbo.gp_utils import kernel
from hyperbo.gp_utils import utils
import jax
import jax.numpy as jnp

DEFAULT_WARP_FUNC = utils.DEFAULT_WARP_FUNC
X_DIM = 5
GPParams = defs.GPParams


class KernelTest(parameterized.TestCase):
  """Tests for kernel.py."""

  @parameterized.named_parameters(
      ('squared_exponential', kernel.squared_exponential,
       GPParams(model={
           'lengthscale': 1.,
           'signal_variance': 1.0,
       })),
      ('matern32', kernel.matern32,
       GPParams(model={
           'lengthscale': 1.,
           'signal_variance': 1.0,
       })),
      ('matern52', kernel.matern52,
       GPParams(model={
           'lengthscale': 1.,
           'signal_variance': 1.0,
       })),
      ('dot_product', kernel.dot_product,
       GPParams(model={
           'dot_prod_sigma': 0.1,
           'dot_prod_bias': 0.1,
       })),
      ('dot_product_mlp', kernel.dot_product_mlp,
       GPParams(
           model={
               'dot_prod_sigma': 0.1,
               'dot_prod_bias': 0.,
           },
           config={
               'mlp_features': (8,),
           })),
      ('squared_exponential_mlp', kernel.squared_exponential_mlp,
       GPParams(
           model={
               'lengthscale': 1.,
               'signal_variance': 1.0,
           },
           config={
               'mlp_features': (8,),
           })),
  )
  def test_kernel_shape(self, cov_func, params):
    """Test gram matrix has the expected shape."""
    key1, key2, key = jax.random.split(jax.random.PRNGKey(0), 3)
    n1, n2 = 10, 15
    vx1 = jax.random.normal(key1, (n1, X_DIM))
    vx2 = jax.random.normal(key2, (n2, X_DIM))
    if 'mlp_features' in params.config:
      bf.init_mlp_with_shape(key, params, vx1.shape)
    logging.info(msg=f'params shapes:\n{jax.tree_map(jnp.shape, params)}')
    logging.info(msg=f'params:\n{params}')
    cov = cov_func(params, vx1, vx2)
    logging.info(msg=f'cov = {cov}')
    self.assertEqual(cov.shape, (n1, n2))

  @parameterized.named_parameters(
      ('squared_exponential', kernel.squared_exponential,
       GPParams(model={
           'lengthscale': 1.,
           'signal_variance': 1.0,
       })),
      ('matern32', kernel.matern32,
       GPParams(model={
           'lengthscale': 1.,
           'signal_variance': 1.0,
       })),
      ('matern52', kernel.matern52,
       GPParams(model={
           'lengthscale': 1.,
           'signal_variance': 1.0,
       })),
      ('dot_product', kernel.dot_product,
       GPParams(model={
           'dot_prod_sigma': 0.1,
           'dot_prod_bias': 0.1,
       })),
      ('dot_product_mlp', kernel.dot_product_mlp,
       GPParams(
           model={
               'dot_prod_sigma': 0.1,
               'dot_prod_bias': 0.,
           },
           config={
               'mlp_features': (8,),
           })),
      ('squared_exponential_mlp', kernel.squared_exponential_mlp,
       GPParams(
           model={
               'lengthscale': 1.,
               'signal_variance': 1.0,
           },
           config={
               'mlp_features': (8,),
           })),
  )
  def test_kernel_psd(self, cov_func, params):
    """Test gram matrix is PSD."""
    logging.log(msg=cov_func.__name__, level=logging.INFO)
    logging.info(msg=f'params = {params}')
    key = jax.random.PRNGKey(0)
    n = 5
    vx = jax.random.normal(key, (n, 5))

    warp_func = DEFAULT_WARP_FUNC
    if 'mlp_features' in params.config:
      bf.init_mlp_with_shape(key, params, vx.shape)

    def check_psd(kfunc):
      gram = kfunc(params, vx, warp_func=warp_func)
      self.assertEqual(gram.shape, (n, n))
      for a, b in zip(gram.flatten(), gram.T.flatten()):
        self.assertAlmostEqual(a, b, places=3)
      eig_vals, _ = jnp.linalg.eig(gram)
      logging.log(msg=f'eig_vals={eig_vals.sort()}', level=logging.INFO)
      self.assertGreaterEqual(eig_vals.min(), 0)

    check_psd(cov_func)


if __name__ == '__main__':
  absltest.main()
