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

"""Common utils for gp_utils."""

from hyperbo.basics import params_utils

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jspla

vmap = jax.vmap
custom_vjp = jax.custom_derivatives.custom_vjp
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


def svd_matrix_sqrt(cov):
  """Compute the square root of a symmetric matrix via SVD.

  Args:
    cov: a symmetric matrix.

  Returns:
    The decomposed factor A such that A * A.T = cov and A is full column rank.
  """
  (u, s, _) = jspla.svd(cov)
  factor = u * jnp.sqrt(s[..., None, :])
  tol = s.max() * jnp.finfo(s.dtype).eps / 2. * jnp.sqrt(2*cov.shape[0] + 1.)
  rank = jnp.count_nonzero(s > tol)
  return factor[:, :rank]


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
