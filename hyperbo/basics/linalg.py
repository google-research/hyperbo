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

from hyperbo.basics import params_utils

import jax
from jax.custom_derivatives import custom_vjp
import jax.numpy as jnp
import jax.scipy.linalg as jspla

vmap = jax.vmap
EPS = 1e-10


def solve_linear_system(coeff, b):
  """Solve linear system Ax = b where A=coeff."""
  chol = jspla.cholesky(coeff, lower=True)
  kinvy = inverse_spdmatrix_vector_product(coeff, b, cached_cholesky=chol)
  return chol, kinvy


def compute_delta_y_and_cov(mean_func,
                            cov_func,
                            params,
                            x,
                            y,
                            warp_func=None,
                            eps=1e-6):
  """Compute y-mu(x) and cov(x,x)+I*sigma^2.

  Args:
    mean_func: mean function handle that maps from (params, n x d input,
      warp_func) to an n dimensional mean vector. (see vector_map in mean.py for
      more details).
    cov_func: covariance function handle that maps from (params, n1 x d input1,
      n2 x d input2, wrap_func) to a n1 x n2  covariance matrix (see matrix_map
      in kernel.py for more details).
    params: parameters for the GP.
    x: n x d dimensional input array for n data points.
    y: n x 1 dimensional evaluation array for n data points.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.
    eps: extra value added to the diagonal of the covariance matrix for
      numerical stability.

  Returns:
    y-mu(x) and cov(x,x)+I*sigma^2.
  """
  y = y - jnp.atleast_2d(mean_func(params, x, warp_func=warp_func))
  noise_variance, = params_utils.retrieve_params(
      params, ['noise_variance'], warp_func=warp_func)
  cov = cov_func(
      params, x, warp_func=warp_func) + jnp.eye(len(x)) * (
          noise_variance + eps)
  return y, cov


def solve_gp_linear_system(mean_func,
                           cov_func,
                           params,
                           x,
                           y,
                           warp_func=None,
                           eps=1e-6):
  """Solve a linear system specified by a GP.

  This function solves m + Kv = y using the Cholesky decomposition of K, where
  K is the covariance matrix with noise terms on the diagonal entries, and m is
  the mean values.

  Args:
    mean_func: mean function handle that maps from (params, n x d input,
      warp_func) to an n dimensional mean vector. (see vector_map in mean.py for
      more details).
    cov_func: covariance function handle that maps from (params, n1 x d input1,
      n2 x d input2, wrap_func) to a n1 x n2  covariance matrix (see matrix_map
      in kernel.py for more details).
    params: parameters for the GP.
    x: n x d dimensional input array for n data points.
    y: n x 1 dimensional evaluation array for n data points.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.
    eps: extra value added to the diagonal of the covariance matrix for
      numerical stability.

  Returns:
    chol: Cholesky decomposition of K (gram matrix).
    kinvy: a convenient intermediate value, K inverse times y, where y has
    already subtracted mean.
    y: y value with mean subtracted.
  """
  y, cov = compute_delta_y_and_cov(
      mean_func, cov_func, params, x, y, warp_func, eps
  )
  chol, kinvy = solve_linear_system(cov, y)
  return chol, kinvy, y


def kl_multivariate_normal(mu0,
                           cov0,
                           mu1,
                           cov1,
                           weight=1.,
                           partial=True,
                           feat0=None,
                           eps=0.):
  """Computes KL divergence between two multivariate normal distributions.

  Args:
    mu0: mean for the first multivariate normal distribution.
    cov0: covariance matrix for the first multivariate normal distribution.
    mu1: mean for the second multivariate normal distribution.
    cov1: covariance matrix for the second multivariate normal distribution.
      cov1 must be invertible.
    weight: weight for the returned KL divergence.
    partial: only compute terms in KL involving mu1 and cov1 if True.
    feat0: (optional) feature used to compute cov0 if cov0 = feat0 * feat0.T /
      feat0.shape[1]. For a low-rank cov0, we may have to compute the KL
      divergence for a degenerate multivariate normal.
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
  chol1, cov1invmudiff = solve_linear_system(cov1, mu_diff)
  # pylint: disable=g-long-lambda
  func = lambda x: inverse_spdmatrix_vector_product(
      cov1, x, cached_cholesky=chol1)
  trcov1invcov0 = jnp.trace(vmap(func)(cov0))
  mahalanobis = jnp.dot(mu_diff, cov1invmudiff)
  logdetcov1 = jnp.sum(2 * jnp.log(jnp.diag(chol1)))
  common_terms = trcov1invcov0 + mahalanobis + logdetcov1
  if partial:
    return 0.5 * weight * common_terms
  else:
    if feat0 is not None and feat0.shape[0] > feat0.shape[1]:
      logging.info('Using pseudo determinant of cov0.')
      sign, logdetcov0 = jnp.linalg.slogdet(
          jnp.divide(jnp.dot(feat0.T, feat0), feat0.shape[1]))
      logging.info(msg=f'Pseudo logdetcov0 = {logdetcov0}')
      assert sign == 1., 'Pseudo determinant of cov0 is 0 or negative.'

      # cov0inv is computed for more accurate pseudo KL. feat0 may be low rank.
      cov0inv = jnp.linalg.pinv(cov0)
      return 0.5 * weight * (
          common_terms - logdetcov0 -
          jnp.linalg.matrix_rank(jnp.dot(cov0inv, cov0)) + jnp.log(2 * jnp.pi) *
          (cov1.shape[0] - feat0.shape[1]))
    else:
      sign, logdetcov0 = jnp.linalg.slogdet(cov0)
      logging.info(msg=f'sign = {sign}; logdetcov0 = {logdetcov0}')
      assert sign == 1., 'Determinant of cov0 is 0 or negative.'
      return 0.5 * weight * (common_terms - logdetcov0 - cov0.shape[0])




def cholesky_cache(spd_matrix, cached_cholesky):
  """Computes the Cholesky factor of `spd_matrix` unless one is already given."""
  if cached_cholesky is not None:
    chol_factor = cached_cholesky
  else:
    chol_factor = jspla.cholesky(spd_matrix, lower=True)

  return chol_factor


@custom_vjp
def inverse_spdmatrix_vector_product(spd_matrix, x, cached_cholesky=None):
  """Computes the inverse matrix vector product where the matrix is SPD."""
  chol_factor = cholesky_cache(spd_matrix, cached_cholesky)

  out = jspla.cho_solve((chol_factor, True), x)
  return out


def inverse_spdmatrix_vector_product_fwd(spd_matrix, x, cached_cholesky=None):
  """Computes the matrix product between the inverse of `spd_matrix` and x."""
  chol_factor = cholesky_cache(spd_matrix, cached_cholesky)

  out = inverse_spdmatrix_vector_product(
      spd_matrix, x, cached_cholesky=chol_factor)
  return out, (chol_factor, x)


def inverse_spdmatrix_vector_product_bwd(res, g):
  """spd_matrix inverse product custom gradient."""
  chol_factor, x = res

  inv_spd_matrix_x = jspla.cho_solve((chol_factor, True), x)
  inv_spd_matrix_g = jspla.cho_solve((chol_factor, True), g)

  grad_spd_matrix = -(jnp.outer(inv_spd_matrix_x, inv_spd_matrix_g))
  grad_x = jspla.cho_solve((chol_factor, True), g)

  return (grad_spd_matrix, grad_x, None)


inverse_spdmatrix_vector_product.defvjp(inverse_spdmatrix_vector_product_fwd,
                                        inverse_spdmatrix_vector_product_bwd)

# An implementation of sqrt that returns a large value for the gradient at 0
# instead of nan.
_safe_sqrt = jax.custom_vjp(jnp.sqrt)


def _safe_sqrt_fwd(x):
  result, vjpfun = jax.vjp(jnp.sqrt, x)
  return result, (x, vjpfun)


def _safe_sqrt_rev(primals, tangent):
  x, vjpfun = primals
  # max_grad = dtypes.finfo(dtypes.dtype(x)).max
  max_grad = 1e6
  result = jnp.where(x != 0., vjpfun(tangent)[0], jnp.full_like(x, max_grad))
  return (result,)


_safe_sqrt.defvjp(_safe_sqrt_fwd, _safe_sqrt_rev)


def safe_l2norm(x):
  """Safe way to compute l2 norm on x without a nan gradient."""
  sqdist = jnp.sum(x**2)
  return _safe_sqrt(sqdist)
