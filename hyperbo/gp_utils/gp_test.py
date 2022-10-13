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
import jax.numpy as jnp

DEFAULT_WARP_FUNC = utils.DEFAULT_WARP_FUNC
GPParams = defs.GPParams

retrieve_params = params_utils.retrieve_params


class GPTest(parameterized.TestCase):
  """Tests for gp.py."""

  @parameterized.named_parameters(
      ('squared_exponential GP.train', kernel.squared_exponential),
      ('matern32 GP.train', kernel.matern32),
      ('matern52 GP.train', kernel.matern52),
      ('squared_exponential infer_parameters', kernel.squared_exponential),
      ('matern32 infer_parameters', kernel.matern32),
      ('matern52 infer_parameters', kernel.matern52),
      ('squared_exponential_mlp GP.train', kernel.squared_exponential_mlp),
      ('dot_product_mlp infer_parameters', kernel.dot_product_mlp),
  )
  def test_infer_parameters(self, cov_func):
    """Test that GP parameters can be inferred correctly."""
    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)
    n = 100
    vx = jax.random.normal(key, (n, 1))
    params = GPParams(
        model={
            'constant': 5.,
            'lengthscale': 1.,
            'signal_variance': 1.0,
            'noise_variance': 0.01,
        })
    if cov_func == kernel.squared_exponential_mlp:
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

    dataset = [(vx, sample_from_gp(i)) for i in range(10)]
    dict_dataset = {}
    for i in range(len(dataset)):
      dict_dataset[i] = defs.SubDataset(*dataset[i])

    def nll_func(gpparams, gpwarp_func=None):
      return obj.neg_log_marginal_likelihood(
          mean_func=mean_func,
          cov_func=cov_func,
          params=gpparams,
          dataset=dict_dataset,
          warp_func=gpwarp_func)

    logging.info(msg=f'NLL on ground truth params = {nll_func(params)}')

    # Given dataset, infer the parameters.
    init_params = GPParams(
        model={
            'constant': 5.1,
            'lengthscale': 0.,
            'signal_variance': 0.,
            'noise_variance': -4.
        })
    if cov_func == kernel.squared_exponential_mlp:
      init_params.config['mlp_features'] = None
    elif cov_func == kernel.dot_product_mlp:
      init_params.model['dot_prod_sigma'] = None
      init_params.model['dot_prod_bias'] = 0.
      init_params.config['mlp_features'] = None

    warp_func = DEFAULT_WARP_FUNC

    start_time = time.time()
    init_params.config.update({
        'method': 'adam',
        'learning_rate': 1e-5,
        'beta': 0.9,
        'max_training_step': 1,
        'logging_interval': 1,
        'batch_size': 100,
    })
    model = gp.GP(
        dataset=dataset,
        mean_func=mean_func,
        cov_func=cov_func,
        params=init_params,
        warp_func=warp_func)
    model.initialize_params(init_key)
    init_nll = nll_func(init_params, warp_func)
    logging.info(msg=f'init_params={model.params}')
    logging.info(msg=f'NLL on init params = {init_nll}')
    inferred_params = model.train()
    logging.info(msg=f'Elapsed training time = {time.time() - start_time}')

    keys = params.model.keys()
    retrieved_inferred_params = dict(
        zip(keys, retrieve_params(inferred_params, keys, warp_func=warp_func)))
    logging.info(msg=f'inferred_params = {retrieved_inferred_params}')

    nll = nll_func(inferred_params, warp_func)
    logging.info(msg=f'NLL on inferred params = {nll} (Before: {init_nll})')

    self.assertGreater(init_nll, nll)

  @parameterized.named_parameters(('nx == 20', 20), ('nx == 0', 0))
  def test_predict(self, nx):
    """Test that a GP posterior has the right shape."""
    key = jax.random.PRNGKey(0)
    x_key, y_key, q_key = jax.random.split(key, 3)
    nq = 10
    vx = jax.random.normal(x_key, (nx, 1))
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
    x_query = jax.random.normal(q_key, (nq, 1))
    model = gp.GP(
        dataset=[(vx, vy)],
        mean_func=mean_func,
        cov_func=cov_func,
        params=params)
    mu_model, var_model = model.predict(
        x_query, full_cov=False, with_noise=True)
    mu, var = gp.predict(
        mean_func, cov_func, params, vx, vy, x_query, full_cov=False)
    self.assertEqual(mu.shape, (nq, 1))
    self.assertEqual(var.shape, (nq, 1))
    self.assertEqual(mu_model.shape, (nq, 1))
    self.assertEqual(var_model.shape, (nq, 1))
    for i in range(len(mu)):
      self.assertAlmostEqual(mu[i], mu_model[i])
    for i in range(len(var)):
      self.assertAlmostEqual(var[i] + params.model['noise_variance'],
                             var_model[i])

    self.assertEqual(model.params.cache[0].needs_update, False)
    self.assertEqual(0 in model.params.cache, True)

    mu_, cov = gp.predict(
        mean_func, cov_func, params, vx, vy, x_query, full_cov=True)
    mu_model_, cov_model = model.predict(
        x_query, full_cov=True, with_noise=True)
    self.assertEqual(mu_.shape, (nq, 1))
    self.assertEqual(cov.shape, (nq, nq))
    self.assertEqual(mu_model_.shape, (nq, 1))
    self.assertEqual(cov_model.shape, (nq, nq))
    for i in range(len(mu)):
      self.assertAlmostEqual(mu[i], mu_[i])
      self.assertAlmostEqual(mu[i], mu_model[i])
    for a, b, c in zip(jnp.diag(cov), var.flatten(), jnp.diag(cov_model)):
      self.assertAlmostEqual(a, b, places=6)
      self.assertAlmostEqual(c, b + params.model['noise_variance'], places=6)
    for i in range(len(cov)):
      for j in range(len(cov[0])):
        if i != j:
          self.assertAlmostEqual(cov[i][j], cov_model[i][j])

  def test_update_dataset(self):
    """Test gp.GP.update_dataset."""
    key = jax.random.PRNGKey(0)
    x_key, q_key = jax.random.split(key, 2)
    n, nq, dim = 3, 5, 2
    vx = jax.random.normal(x_key, (n, dim))
    params = GPParams(
        model={
            'constant': 5.,
            'lengthscale': 1.,
            'signal_variance': 1.0,
            'noise_variance': 0.01,
        })
    mean_func = mean.constant
    cov_func = kernel.squared_exponential

    def sample_from_gp(seed):
      return gp.sample_from_gp(
          jax.random.PRNGKey(seed), mean_func, cov_func, params, vx)

    gp_model = gp.GP(
        dataset=[], mean_func=mean_func, cov_func=cov_func, params=params)
    self.assertEqual(gp_model.dataset, {})
    self.assertEqual(gp_model.params.cache, {})
    x_query = jax.random.normal(q_key, (nq, dim))
    mu_model, var_model = gp_model.predict(
        x_query, full_cov=False, with_noise=True)

    self.assertEqual(mu_model.shape, (nq, 1))
    self.assertEqual(var_model.shape, (nq, 1))
    self.assertEqual(gp_model.dataset, {})
    self.assertEqual(gp_model.params.cache, {})

    dataset = [(vx, sample_from_gp(i)) for i in range(10)]

    for i in range(len(dataset)):
      gp_model.update_sub_dataset(dataset[i], i, is_append=False)
      self.assertEqual(gp_model.dataset[i].x.shape, dataset[i][0].shape)
      self.assertEqual(gp_model.dataset[i].y.shape, dataset[i][1].shape)
    self.assertEqual(len(gp_model.dataset), len(dataset))
    self.assertEqual(gp_model.params.cache, {})
    gp_model.update_sub_dataset(dataset[0], 0, is_append=True)
    self.assertEqual(gp_model.dataset[0].x.shape[0], dataset[0][0].shape[0] * 2)
    self.assertEqual(gp_model.dataset[0].y.shape[0], dataset[0][1].shape[0] * 2)

    self.assertEqual('new' in gp_model.dataset, False)
    gp_model.update_sub_dataset(
        (gp_model.dataset[0].x[0], gp_model.dataset[0].y[0]),
        'new',
        is_append=True)
    self.assertEqual('new' in gp_model.dataset, True)
    self.assertEqual(gp_model.dataset['new'].x.shape, (1, dim))
    self.assertEqual(gp_model.dataset['new'].y.shape, (1, 1))

    self.assertEqual('new2' in gp_model.dataset, False)
    gp_model.update_sub_dataset(gp_model.dataset[0], 'new2', is_append=True)
    self.assertEqual('new2' in gp_model.dataset, True)
    self.assertEqual(gp_model.dataset['new2'].x.shape,
                     gp_model.dataset[0].x.shape)
    self.assertEqual(gp_model.dataset['new2'].y.shape,
                     gp_model.dataset[0].y.shape)

    mu_model, var_model = gp_model.predict(
        x_query, sub_dataset_key=5, full_cov=False, with_noise=True)

    self.assertEqual(mu_model.shape, (nq, 1))
    self.assertEqual(var_model.shape, (nq, 1))
    self.assertEqual(5 in gp_model.params.cache, True)
    self.assertEqual(0 in gp_model.params.cache, False)

  def test_sample_from_gp_shape(self):
    """Test that a GP sample has the right shape."""
    key = jax.random.PRNGKey(0)
    x_key, y_key = jax.random.split(key, 2)
    nx, num_samples = 20, 10
    vx = jax.random.normal(x_key, (nx, 1))
    params = GPParams(
        model={
            'constant': 5.,
            'lengthscale': 1.,
            'signal_variance': 1.0,
            'noise_variance': 0.01,
        })
    mean_func = mean.constant
    cov_func = kernel.squared_exponential
    for method in ['svd', 'cholesky']:
      vy = gp.sample_from_gp(
          y_key,
          mean_func,
          cov_func,
          params,
          vx,
          num_samples=num_samples,
          method=method)
      self.assertEqual(vy.shape, (nx, num_samples))


if __name__ == '__main__':
  absltest.main()
