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

"""Test gp.py.

Use the following to debug nan.

from jax.config import config
config.update('jax_debug_nans', True)
"""
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
from hyperbo.gp_utils import priors
from hyperbo.gp_utils import utils
import jax
import jax.numpy as jnp

DEFAULT_WARP_FUNC = utils.DEFAULT_WARP_FUNC
GPParams = defs.GPParams
retrieve_params = params_utils.retrieve_params


class GPTest(parameterized.TestCase):
  """Tests for slice sampling method in gp.py."""

  @parameterized.named_parameters(
      ('squared_exponential', kernel.squared_exponential),
      ('matern32', kernel.matern32),
      ('matern52', kernel.matern52),
      ('matern32_mlp', kernel.matern32_mlp),
      ('matern52_mlp', kernel.matern52_mlp),
      ('squared_exponential_mlp', kernel.squared_exponential_mlp),
      ('dot_product_mlp', kernel.dot_product_mlp),
  )
  def test_slice_sampling(self, cov_func):
    """Test that GP parameters can be inferred correctly."""
    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    n, nq = 6, 3
    vx = jax.random.normal(key, (n, 2))
    key, _ = jax.random.split(key)
    qx = jax.random.normal(key, (nq, 2))
    params = GPParams(
        model={
            'constant': 5.,
            'lengthscale': .1,
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
      params.model['dot_prod_sigma'] = 0.5
      params.model['dot_prod_bias'] = 0.
      params.config['mlp_features'] = (8,)
      key, _ = jax.random.split(key)
      bf.init_mlp_with_shape(key, params, vx.shape)

    mean_func = mean.constant
    logging.info(msg=f'params = {params}')

    def sample_from_gp(seed):
      return gp.sample_from_gp(
          jax.random.PRNGKey(seed), mean_func, cov_func, params, vx)

    dataset = [(vx, sample_from_gp(i)) for i in range(10)]

    # Minimize sample_mean_cov_regularizer.
    nsamples = 1
    init_params = GPParams(
        model={
            'constant': 5.1,
            'lengthscale': jnp.array([0., 0.]),
            'signal_variance': 0.,
            'noise_variance': -4.
        },
        config={
            'method': 'slice_sample',
            'burnin': nsamples,
            'nsamples': nsamples,
            'max_training_step': 0,
            'logging_interval': 1,
            'priors': priors.DEFAULT_PRIORS,
            'mlp_features': (8,),
            'batch_size': 100,
        })
    init_key, _ = jax.random.split(init_key)
    # bf.init_kumar_warp_with_shape(init_key, init_params, vx.shape)
    if cov_func in [
        kernel.squared_exponential_mlp, kernel.matern32_mlp, kernel.matern52_mlp
    ]:
      init_params.model['lengthscale'] = jnp.array([0.] * 8)
    elif cov_func == kernel.dot_product_mlp:
      init_params.model['dot_prod_sigma'] = 1.
      init_params.model['dot_prod_bias'] = 0.
    warp_func = DEFAULT_WARP_FUNC

    model = gp.HGP(
        dataset=dataset,
        mean_func=mean.linear_mlp,
        cov_func=cov_func,
        params=init_params,
        warp_func=warp_func)
    model.initialize_params(init_key)

    init_nll, _, _, _ = model.stats()

    start_time = time.time()
    logging.info(msg=f'init_params={init_params}')
    inferred_params = model.train()
    logging.info(msg=f'Elapsed training time = {time.time() - start_time}')

    inferred_nll, _, _, _ = model.stats()

    keys = params.model.keys()
    retrieved_inferred_params = dict(
        zip(keys, retrieve_params(inferred_params, keys, warp_func=warp_func)))
    logging.info(msg=f'params.model = {retrieved_inferred_params}')

    self.assertGreater(init_nll, inferred_nll)

    predictions = model.predict(qx, 0, True, True)
    logging.info(msg=f'predictions = {predictions}')
    self.assertLen(predictions, nsamples * 2)
    for i in range(nsamples * 2):
      self.assertEqual(predictions[i][0].shape, (nq, 1))
      self.assertEqual(predictions[i][1].shape, (nq, nq))


if __name__ == '__main__':
  absltest.main()
