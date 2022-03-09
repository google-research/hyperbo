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

"""Test gp.py.

Use the following to debug nan.

from jax.config import config
config.update('jax_debug_nans', True)
"""
import functools
import logging
import time

from absl.testing import absltest
from absl.testing import parameterized
from hyperbo.basics import definitions as defs
from hyperbo.basics import params_utils
from hyperbo.gp_utils import basis_functions as bf
from hyperbo.gp_utils import gp
from hyperbo.gp_utils import kernel
from hyperbo.gp_utils import mean
from hyperbo.gp_utils import objectives as obj
from hyperbo.gp_utils import utils
import jax

DEFAULT_WARP_FUNC = utils.DEFAULT_WARP_FUNC
GPParams = defs.GPParams
retrieve_params = params_utils.retrieve_params


class GPTest(parameterized.TestCase):
  """Tests for gp.py."""

  @parameterized.named_parameters(
      ('squared_exponential kl', kernel.squared_exponential,
       utils.kl_multivariate_normal),
      ('matern32 kl', kernel.matern32, utils.kl_multivariate_normal),
      ('matern52 kl', kernel.matern52, utils.kl_multivariate_normal),
      ('matern32_mlp kl', kernel.matern32_mlp, utils.kl_multivariate_normal),
      ('matern52_mlp kl', kernel.matern52_mlp, utils.kl_multivariate_normal),
      ('squared_exponential_mlp kl', kernel.squared_exponential_mlp,
       utils.kl_multivariate_normal),
      ('dot_product_mlp kl', kernel.dot_product_mlp,
       utils.kl_multivariate_normal),
      ('squared_exponential euclidean', kernel.squared_exponential,
       utils.euclidean_multivariate_normal),
      ('matern32 euclidean', kernel.matern32,
       utils.euclidean_multivariate_normal),
      ('matern52 euclidean', kernel.matern52,
       utils.euclidean_multivariate_normal),
      ('matern32_mlp euclidean', kernel.matern32_mlp,
       utils.euclidean_multivariate_normal),
      ('matern52_mlp euclidean', kernel.matern52_mlp,
       utils.euclidean_multivariate_normal),
      ('squared_exponential_mlp euclidean', kernel.squared_exponential_mlp,
       utils.euclidean_multivariate_normal),
      ('dot_product_mlp euclidean', kernel.dot_product_mlp,
       utils.euclidean_multivariate_normal),
  )
  def test_sample_mean_cov_regularizer_split_data(self, cov_func, distance):
    """Test that GP parameters can be inferred correctly."""
    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    n = 20
    vx = jax.random.normal(key, (n, 1))
    params = GPParams(
        model={
            'constant': 5.,
            'lengthscale': 1.,
            'signal_variance': 1.0,
            'noise_variance': 0.01,
        })
    if cov_func in [
        kernel.squared_exponential_mlp, kernel.matern32_mlp, kernel.matern52_mlp
    ]:
      params.config['mlp_features'] = (8,)
      bf.init_mlp_with_shape(key, params, vx.shape)
    elif cov_func == kernel.dot_product_mlp:
      params.model['dot_prod_sigma'] = jax.random.normal(key, (8, 8 * 2))
      params.model['dot_prod_bias'] = 0.
      params.config['mlp_features'] = (8,)
      bf.init_mlp_with_shape(key, params, vx.shape)
    mean_func = mean.constant
    logging.info(msg=f'params = {params}')

    def sample_from_gp(seed):
      return gp.sample_from_gp(
          jax.random.PRNGKey(seed), mean_func, cov_func, params, vx)

    dataset = [(vx, sample_from_gp(i), 0) for i in range(10)]

    # Minimize sample_mean_cov_regularizer.
    init_params = GPParams(
        model={
            'constant': 5.1,
            'lengthscale': 0.,
            'signal_variance': 0.,
            'noise_variance': -4.
        },
        config={
            'method':
                'lbfgs',
            'maxiter':
                1,
            'objective':
                functools.partial(
                    obj.sample_mean_cov_regularizer_split_data,
                    distance=distance)
        })
    if cov_func in [
        kernel.squared_exponential_mlp, kernel.matern32_mlp, kernel.matern52_mlp
    ]:
      init_params.config['mlp_features'] = (8,)
      bf.init_mlp_with_shape(init_key, init_params, vx.shape)
    elif cov_func == kernel.dot_product_mlp:
      init_params.model['dot_prod_sigma'] = jax.random.normal(
          init_key, (8, 8 * 2))
      init_params.model['dot_prod_bias'] = 0.
      init_params.config['mlp_features'] = (8,)
      bf.init_mlp_with_shape(init_key, init_params, vx.shape)

    warp_func = DEFAULT_WARP_FUNC

    model = gp.GP(
        dataset=dataset,
        mean_func=mean_func,
        cov_func=cov_func,
        params=init_params,
        warp_func=warp_func)

    def reg(gpparams, gpwarp_func=None):
      return obj.sample_mean_cov_regularizer_split_data(
          mean_func=model.mean_func,
          cov_func=model.cov_func,
          params=gpparams,
          dataset=model.dataset,
          warp_func=gpwarp_func,
          distance=distance)

    def nll_func(gpparams, gpwarp_func=None):
      return obj.neg_log_marginal_likelihood(
          mean_func=model.mean_func,
          cov_func=model.cov_func,
          params=gpparams,
          dataset=model.dataset,
          warp_func=gpwarp_func,
          exclude_aligned=False)

    logging.info(msg=f'Regularizer on ground truth params = {reg(params)}')
    logging.info(msg=f'NLL on ground truth params = {nll_func(params)}')

    init_reg = reg(init_params, warp_func)
    init_nll = nll_func(init_params, warp_func)
    logging.info(msg=f'Reg on init params = {init_reg}')
    logging.info(msg=f'NLL on init params = {init_nll}')

    start_time = time.time()
    logging.info(msg=f'init_params={init_params}')
    inferred_params = model.train()
    logging.info(msg=f'Elapsed training time = {time.time() - start_time}')

    keys = params.model.keys()
    retrieved_inferred_params = dict(
        zip(keys, retrieve_params(inferred_params, keys, warp_func=warp_func)))
    logging.info(msg=f'inferred_params = {retrieved_inferred_params}')

    inferred_reg = reg(inferred_params, warp_func)
    inferred_nll = nll_func(inferred_params, warp_func)
    logging.info(
        msg=f'Reg on inferred params = {inferred_reg} (Before: {init_reg})')
    logging.info(
        msg=f'NLL on inferred params = {inferred_nll} (Before: {init_nll})')

    self.assertGreater(init_reg, inferred_reg)

  @parameterized.named_parameters(
      ('squared_exponential kl', kernel.squared_exponential,
       utils.kl_multivariate_normal),
      ('matern32 kl', kernel.matern32, utils.kl_multivariate_normal),
      ('matern52 kl', kernel.matern52, utils.kl_multivariate_normal),
      ('matern32_mlp kl', kernel.matern32_mlp, utils.kl_multivariate_normal),
      ('matern52_mlp kl', kernel.matern52_mlp, utils.kl_multivariate_normal),
      ('squared_exponential_mlp kl', kernel.squared_exponential_mlp,
       utils.kl_multivariate_normal),
      ('dot_product_mlp kl', kernel.dot_product_mlp,
       utils.kl_multivariate_normal),
      ('squared_exponential euclidean', kernel.squared_exponential,
       utils.euclidean_multivariate_normal),
      ('matern32 euclidean', kernel.matern32,
       utils.euclidean_multivariate_normal),
      ('matern52 euclidean', kernel.matern52,
       utils.euclidean_multivariate_normal),
      ('matern32_mlp euclidean', kernel.matern32_mlp,
       utils.euclidean_multivariate_normal),
      ('matern52_mlp euclidean', kernel.matern52_mlp,
       utils.euclidean_multivariate_normal),
      ('squared_exponential_mlp euclidean', kernel.squared_exponential_mlp,
       utils.euclidean_multivariate_normal),
      ('dot_product_mlp euclidean', kernel.dot_product_mlp,
       utils.euclidean_multivariate_normal),
  )
  def test_sample_mean_cov_regularizer(self, cov_func, distance):
    """Test that GP parameters can be inferred correctly."""
    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    n = 20
    vx = jax.random.normal(key, (n, 2))
    params = GPParams(
        model={
            'constant': 5.,
            'lengthscale': 1.,
            'signal_variance': 1.0,
            'noise_variance': 0.01,
        })
    if cov_func in [
        kernel.squared_exponential_mlp, kernel.matern32_mlp, kernel.matern52_mlp
    ]:
      params.config['mlp_features'] = (8,)
      key, _ = jax.random.split(key)
      bf.init_mlp_with_shape(key, params, vx.shape)
    elif cov_func == kernel.dot_product_mlp:
      key, _ = jax.random.split(key)
      params.model['dot_prod_sigma'] = jax.random.normal(key, (8, 8 * 2))
      params.model['dot_prod_bias'] = 0.
      params.config['mlp_features'] = (8,)
      key, _ = jax.random.split(key)
      bf.init_mlp_with_shape(key, params, vx.shape)

    mean_func = mean.constant
    logging.info(msg=f'params = {params}')

    key, _ = jax.random.split(key)
    dataset = [(vx,
                gp.sample_from_gp(
                    key, mean_func, cov_func, params, vx,
                    num_samples=10), 'all_data')]

    # Minimize sample_mean_cov_regularizer.
    init_params = GPParams(
        model={
            'constant': 5.1,
            'lengthscale': 0.,
            'signal_variance': 0.,
            'noise_variance': -4.
        },
        config={
            'method':
                'lbfgs',
            'maxiter':
                1,
            'objective':
                functools.partial(
                    obj.sample_mean_cov_regularizer, distance=distance)
        })
    if cov_func in [
        kernel.squared_exponential_mlp, kernel.matern32_mlp, kernel.matern52_mlp
    ]:
      init_params.config['mlp_features'] = (8,)
      bf.init_mlp_with_shape(init_key, init_params, vx.shape)
    elif cov_func == kernel.dot_product_mlp:
      init_params.model['dot_prod_sigma'] = jax.random.normal(
          init_key, (8, 8 * 2))
      init_params.model['dot_prod_bias'] = 0.
      init_params.config['mlp_features'] = (8,)
      bf.init_mlp_with_shape(init_key, init_params, vx.shape)

    warp_func = DEFAULT_WARP_FUNC

    model = gp.GP(
        dataset=dataset,
        mean_func=mean_func,
        cov_func=cov_func,
        params=init_params,
        warp_func=warp_func)

    def reg(gpparams, gpwarp_func=None):
      return obj.sample_mean_cov_regularizer(
          mean_func=model.mean_func,
          cov_func=model.cov_func,
          params=gpparams,
          dataset=model.dataset,
          warp_func=gpwarp_func,
          distance=distance)

    def nll_func(gpparams, gpwarp_func=None):
      return obj.neg_log_marginal_likelihood(
          mean_func=model.mean_func,
          cov_func=model.cov_func,
          params=gpparams,
          dataset=model.dataset,
          warp_func=gpwarp_func)

    logging.info(msg=f'Regularizer on ground truth params = {reg(params)}')
    logging.info(msg=f'NLL on ground truth params = {nll_func(params)}')

    init_reg = reg(init_params, warp_func)
    init_nll = nll_func(init_params, warp_func)
    logging.info(msg=f'Reg on init params = {init_reg}')
    logging.info(msg=f'NLL on init params = {init_nll}')

    start_time = time.time()
    logging.info(msg=f'init_params={init_params}')
    inferred_params = model.train()
    logging.info(msg=f'Elapsed training time = {time.time() - start_time}')

    keys = params.model.keys()
    retrieved_inferred_params = dict(
        zip(keys, retrieve_params(inferred_params, keys, warp_func=warp_func)))
    logging.info(msg=f'inferred_params = {retrieved_inferred_params}')

    inferred_reg = reg(inferred_params, warp_func)
    inferred_nll = nll_func(inferred_params, warp_func)
    logging.info(
        msg=f'Reg on inferred params = {inferred_reg} (Before: {init_reg})')
    logging.info(
        msg=f'NLL on inferred params = {inferred_nll} (Before: {init_nll})')

    self.assertGreater(init_reg, inferred_reg)


if __name__ == '__main__':
  absltest.main()
