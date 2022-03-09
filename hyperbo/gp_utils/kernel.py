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

"""Kernel library for GP. GPML book: http://www.gaussianprocess.org/gpml/."""
import functools

from hyperbo.basics import linalg
from hyperbo.basics import params_utils
from hyperbo.gp_utils import basis_functions as bf
import jax
import jax.numpy as jnp

retrieve_params = params_utils.retrieve_params
vmap = jax.vmap


def covariance_matrix(kernel):
  """Decorator to kernels to obtain the covariance matrix."""

  @functools.wraps(kernel)
  def matrix_map(params, vx1, vx2=None, warp_func=None, diag=False):
    """Returns the kernel matrix of input array vx1 and input array vx2.

    Args:
      params: parameters for the kernel.
      vx1: n1 x d dimensional input array representing n1 data points.
      vx2: n2 x d dimensional input array representing n2 data points. If it is
        not specified, vx2 is set to be the same as vx1.
      warp_func: optional dictionary that specifies the warping function for
        each parameter.
      diag: flag for returning diagonal terms of the matrix (True) or the full
        matrix (False).

    Returns:
      The n1 x n2 dimensional covariance matrix derived from kernel evaluations
        on every pair of inputs from vx1 and vx2 by default. If diag=True and
        vx2=None, it returns the diagonal terms of the n1 x n1 covariance
        matrix.
    """
    cov_func = functools.partial(kernel, params, warp_func=warp_func)
    mmap = vmap(lambda x: vmap(lambda y: cov_func(x, y))(vx1))
    if vx2 is None:
      if diag:
        return vmap(lambda x: cov_func(x, x))(vx1)
      vx2 = vx1
    return mmap(vx2).T

  return matrix_map


@covariance_matrix
def squared_exponential(params, x1, x2, warp_func=None):
  """Squared exponential kernel: Eq.(4.9/13) of GPML book.

  Args:
    params: parameters for the kernel.
    x1: a d-diemnsional vector that represent a single datapoint.
    x2: a d-diemnsional vector that represent a single datapoint that can be the
      same as or different from x1.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.

  Returns:
    The kernel function evaluation on x1 and x2.
  """
  params_keys = ['lengthscale', 'signal_variance']
  lengthscale, signal_variance = retrieve_params(params, params_keys, warp_func)
  r2 = jnp.sum(((x1 - x2) / lengthscale)**2)
  return jnp.squeeze(signal_variance) * jnp.exp(-r2 / 2)


@covariance_matrix
def matern32(params, x1, x2, warp_func=None):
  """Matern 3/2 kernel: Eq.(4.17) of GPML book.

  Args:
    params: parameters for the kernel.
    x1: a d-diemnsional vector that represent a single datapoint.
    x2: a d-diemnsional vector that represent a single datapoint that can be the
      same as or different from x1.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.

  Returns:
    The kernel function evaluation on x1 and x2.
  """
  params_keys = ['lengthscale', 'signal_variance']
  lengthscale, signal_variance = retrieve_params(params, params_keys, warp_func)
  r = jnp.sqrt(3) * linalg.safe_l2norm((x1 - x2) / lengthscale)
  return jnp.squeeze(signal_variance) * (1 + r) * jnp.exp(-r)


@covariance_matrix
def matern52(params, x1, x2, warp_func=None):
  """Matern 5/2 kernel: Eq.(4.17) of GPML book.

  Args:
    params: parameters for the kernel.
    x1: a d-diemnsional vector that represent a single datapoint.
    x2: a d-diemnsional vector that represent a single datapoint that can be the
      same as or different from x1.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.

  Returns:
    The kernel function evaluation on x1 and x2.
  """
  params_keys = ['lengthscale', 'signal_variance']
  lengthscale, signal_variance = retrieve_params(params, params_keys, warp_func)
  r = jnp.sqrt(5) * linalg.safe_l2norm((x1 - x2) / lengthscale)
  return signal_variance * (1 + r + r**2 / 3) * jnp.exp(-r)


@covariance_matrix
def dot_product(params, x1, x2, warp_func=None):
  r"""Dot product kernel.

  Args:
    params: parameters for the kernel. S=params['dot_prod_sigma'] and
      B=params['dot_prod_bias']corresponds to \Sigma_p=SS^T, \sigma_0=B in
      Section 4.2.2 of GPML book. The kernel is k(x, x') = \sigma_0^2 +
      x^T\Sigma_p x'. S is d x d' (d' >= d) and B is a float.
    x1: a d-diemnsional vector that represent a single datapoint.
    x2: a d-diemnsional vector that represent a single datapoint that can be the
      same as or different from x1.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.

  Returns:
    The kernel function evaluation on x1 and x2.
  """
  params_keys = ['dot_prod_sigma', 'dot_prod_bias']
  sigma, bias = retrieve_params(params, params_keys, warp_func)
  return jnp.dot(jnp.dot(x1, jnp.dot(sigma, sigma.T)), x2.T) + jnp.square(bias)


def with_mlp_bases(kernel):
  """Wrapper for kernels to obtain the covariance matrix on MLP bases."""

  def kernel_mlp(params, vx1, vx2=None, warp_func=None, diag=False):
    """Returns the kernel matrix with MLP basis functions.

    Args:
      params: parameters for the kernel.
      vx1: n1 x d dimensional input array representing n1 data points.
      vx2: n2 x d dimensional input array representing n2 data points. If it is
        not specified, vx2 is set to be the same as vx1.
      warp_func: optional dictionary that specifies the warping function for
        each parameter.
      diag: flag for returning diagonal terms of the matrix (True) or the full
        matrix (False).

    Returns:
      The n1 x n2 dimensional covariance matrix derived from kernel evaluations
        on every pair of inputs from vx1 and vx2 by default. If diag=True and
        vx2=None, it returns the diagonal terms of the n1 x n1 covariance
        matrix.
    """
    model = bf.MLP(params.config['mlp_features'])
    mlp_params, = retrieve_params(params, ['mlp_params'], warp_func)
    vx1 = model.apply({'params': mlp_params}, vx1)
    if vx2 is not None:
      vx2 = model.apply({'params': mlp_params}, vx2)
    return kernel(params, vx1, vx2, warp_func=warp_func, diag=diag)

  return kernel_mlp


dot_product_mlp = with_mlp_bases(dot_product)
squared_exponential_mlp = with_mlp_bases(squared_exponential)
matern32_mlp = with_mlp_bases(matern32)
matern52_mlp = with_mlp_bases(matern52)


def with_kumar_bases(kernel):
  """Decorator to kernels to obtain the cov matrix on Kumaraswamy bases."""

  @functools.wraps(kernel)
  def matrix_map(params, vx1, vx2=None, warp_func=None, diag=False):
    """Returns the kernel matrix with MLP basis functions.

    Args:
      params: parameters for the kernel.
      vx1: n1 x d dimensional input array representing n1 data points.
      vx2: n2 x d dimensional input array representing n2 data points. If it is
        not specified, vx2 is set to be the same as vx1.
      warp_func: optional dictionary that specifies the warping function for
        each parameter.
      diag: flag for returning diagonal terms of the matrix (True) or the full
        matrix (False).

    Returns:
      The n1 x n2 dimensional covariance matrix derived from kernel evaluations
        on every pair of inputs from vx1 and vx2 by default. If diag=True and
        vx2=None, it returns the diagonal terms of the n1 x n1 covariance
        matrix.
    """
    model = bf.KumarWarp()
    kumar_params, = retrieve_params(params, ['kumar_params'], warp_func)
    vx1 = model.apply({'params': kumar_params}, vx1)
    if vx2 is not None:
      vx2 = model.apply({'params': kumar_params}, vx2)
    return kernel(params, vx1, vx2, warp_func=warp_func, diag=diag)

  return matrix_map


dot_product_kumar = with_kumar_bases(dot_product)
squared_exponential_kumar = with_kumar_bases(squared_exponential)
matern32_kumar = with_kumar_bases(matern32)
matern52_kumar = with_kumar_bases(matern52)
