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

"""Common utils for gp_utils."""

import logging

from hyperbo.basics import definitions as defs
from hyperbo.basics import linalg
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jspla

# import jax._src.dtypes as dtypes

SubDataset = defs.SubDataset

vmap = jax.vmap
EPS = 1e-10

identity_warp = lambda x: x
softplus_warp = jax.nn.softplus


def sub_sample_dataset_iterator(key, dataset, batch_size):
  """Iterator for subsample a dataset such that each sub_dataset has at most batch_size data points.

  Args:
    key: Jax random state.
    dataset: dict of SubDataset.
    batch_size: int, maximum number of data points per sub dataset in a batch.

  Yields:
    A sub sampled dataset batch.
  """
  while True:
    sub_sampled_dataset = {}
    for sub_dataset_key, sub_dataset in dataset.items():
      if sub_dataset.x.shape[0] >= batch_size:
        key, subkey = jax.random.split(key, 2)
        indices = jax.random.permutation(subkey, sub_dataset.x.shape[0])
        ifaligned = sub_dataset.aligned
        sub_sampled_dataset[sub_dataset_key] = SubDataset(
            x=sub_dataset.x[indices[:batch_size], :],
            y=sub_dataset.y[indices[:batch_size], :],
            aligned=ifaligned)
      else:
        sub_sampled_dataset[sub_dataset_key] = sub_dataset
    yield sub_sampled_dataset


def squareplus_warp(x):
  """Alternative to softplus with nicer properties.

  See https://twitter.com/jon_barron/status/1387167648669048833

  Args:
    x: scalar or numpy array.

  Returns:
    The transformed x.
  """
  return 0.5 * (x + jnp.sqrt(x**2 + 4))


DEFAULT_SOFTPLUS = lambda x: softplus_warp(x) + EPS

DEFAULT_WARP_FUNC = {
    'constant': identity_warp,
    'lengthscale': DEFAULT_SOFTPLUS,
    'signal_variance': DEFAULT_SOFTPLUS,
    'noise_variance': DEFAULT_SOFTPLUS,
    'dot_prod_sigma': DEFAULT_SOFTPLUS,
}


def svd_matrix_sqrt(cov):
  """Compute the square root of a symmetric matrix via SVD.

  Args:
    cov: a symmetric matrix.

  Returns:
    The decomposed factor A such that A * A.T = cov and A is full column rank.
  """
  (u, s, _) = jspla.svd(cov)
  factor = u * jnp.sqrt(s[..., None, :])
  tol = (
      s.max() * jnp.finfo(s.dtype).eps / 2.0 * jnp.sqrt(2 * cov.shape[0] + 1.0)
  )
  rank = jnp.count_nonzero(s > tol)
  return factor[:, :rank]


def kl_multivariate_normal(
    mu0, cov0, mu1, cov1, weight=1.0, partial=True, eps=0.0
):
  """Computes KL divergence between two multivariate normal distributions.

  Args:
    mu0: mean for the first multivariate normal distribution.
    cov0: covariance matrix for the first multivariate normal distribution.
    mu1: mean for the second multivariate normal distribution.
    cov1: covariance matrix for the second multivariate normal distribution.
      cov1 must be invertible.
    weight: weight for the returned KL divergence.
    partial: only compute terms in KL involving mu1 and cov1 if True.
    eps: (optional) small positive value added to the diagonal terms of cov0 and
      cov1 to make them well behaved.

  Returns:
    KL divergence. The returned value does not include terms that are not
    affected by potential model parameters in mu1 or cov1.
  """
  if not cov0.shape:
    cov0 = cov0[jnp.newaxis, jnp.newaxis]
  if not cov1.shape:
    cov1 = cov1[jnp.newaxis, jnp.newaxis]

  if eps > 0.:
    cov0 = cov0 + jnp.eye(cov0.shape[0]) * eps
    cov1 = cov1 + jnp.eye(cov1.shape[0]) * eps

  mu_diff = mu1 - mu0
  chol1, cov1invmudiff = linalg.solve_linear_system(cov1, mu_diff)
  # pylint: disable=g-long-lambda
  func = lambda x: linalg.inverse_spdmatrix_vector_product(
      cov1, x, cached_cholesky=chol1
  )
  trcov1invcov0 = jnp.trace(vmap(func)(cov0))
  mahalanobis = jnp.dot(mu_diff, cov1invmudiff)
  common_terms = trcov1invcov0 + mahalanobis
  if partial:
    cov0_sqrt = svd_matrix_sqrt(cov0)
    if cov0_sqrt.shape[1] < cov0_sqrt.shape[1]:
      sign, logdetcov1 = jnp.linalg.slogdet(
          jnp.dot(cov0_sqrt.T, jnp.dot(cov1, cov0_sqrt))
      )
      logging.info(msg=f'sign, logdetcov1 = {sign}, {logdetcov1}')
      assert sign == 1.0, 'Determinant of cov1 is 0 or negative.'
    else:
      logdetcov1 = jnp.sum(2 * jnp.log(jnp.diag(chol1)))
    ekl = common_terms + logdetcov1
    return weight * ekl
  # Compute the full EKL
  cov0_sqrt = svd_matrix_sqrt(cov0)
  sign, logdetcov0 = jnp.linalg.slogdet(jnp.dot(cov0_sqrt.T, cov0_sqrt))
  logging.info(msg=f'sign, logdetcov0 = {sign}, {logdetcov0}')
  assert sign == 1., 'Pseudo determinant of cov0 is 0 or negative.'
  sign, logdetcov1 = jnp.linalg.slogdet(
      jnp.dot(cov0_sqrt.T, jnp.dot(cov1, cov0_sqrt))
  )
  logging.info(msg=f'sign, logdetcov1 = {sign}, {logdetcov1}')
  assert sign == 1., 'Determinant of cov1 is 0 or negative.'
  ekl = 0.5 * (common_terms + logdetcov1 - 2 * logdetcov0 - cov0_sqrt.shape[1])
  return weight * ekl


def euclidean_multivariate_normal(mu0,
                                  cov0,
                                  mu1,
                                  cov1,
                                  mean_weight=1.,
                                  cov_weight=1.,
                                  **unused_kwargs):
  """Computes Euclidean distance between two multivariate normal distributions.

  Args:
    mu0: mean for the first multivariate normal distribution.
    cov0: covariance matrix for the first multivariate normal distribution.
    mu1: mean for the second multivariate normal distribution.
    cov1: covariance matrix for the second multivariate normal distribution.
    mean_weight: weight for euclidean distance on the mean vectors.
    cov_weight: weight for euclidean distance on the covariance matrices.

  Returns:
    Reweighted Euclidean distance between two multivariate normal distributions.
  """
  mean_diff = linalg.safe_l2norm(mu0 - mu1)
  cov_diff = linalg.safe_l2norm((cov0 - cov1).flatten())
  return mean_weight * mean_diff + cov_weight * cov_diff
