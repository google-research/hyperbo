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

"""Noise variance function library for GP."""
import functools

from flax import linen as nn
from hyperbo.basics import params_utils
from hyperbo.gp_utils import basis_functions as bf
import jax
import jax.numpy as jnp

EPS = 1e-10
vmap = jax.vmap

retrieve_params = params_utils.retrieve_params

# Recommended warping functions for noise variance to ensure positivity.
softplus_warp = lambda x: jnp.logaddexp(x, 0.) + EPS
squared_warp = lambda x: jnp.square(x) + EPS


def get_noise_variance(val, warp_func=None):
  """Get the noise variance value after warping.

  Args:
    val: pre-warped noise variance value.
    warp_func: ptional dictionary that specifies the warping function for
        each parameter.

  Returns:
    Noise variance.
  """
  noise_variance = val
  if warp_func and 'noise_variance' in warp_func:
    noise_variance_warp = warp_func['noise_variance']
    noise_variance = noise_variance_warp(noise_variance)
  return noise_variance


def noise_variance_vector(noise_variance_func):
  """Decorator to noise_variance functions to obtain the noise_variance vector."""

  @functools.wraps(noise_variance_func)
  def vector_map(params, vx, warp_func=None):
    """Returns the noise_variance vector of input array vx.

    Args:
      params: parameters for the noise_variance function.
      vx: n x d dimensional input array representing n data points.
      warp_func: optional dictionary that specifies the warping function for
        each parameter.

    Returns:
      The n dimensional noise_variance (column) vector derived from
      noise_variance
      function evaluations
        on every input from vx.
    """
    # pylint: disable=unnecessary-lambda,g-long-lambda
    parameterized_noise_variance_func = lambda x: noise_variance_func(
        params, x, warp_func
    )
    return vmap(parameterized_noise_variance_func)(vx)

  return vector_map


@noise_variance_vector
def constant(params, _, warp_func=None):
  """Constant noise_variance function."""
  (val,) = retrieve_params(params, ['constant_noise_variance'], warp_func)
  return get_noise_variance(jnp.full((1,), val), warp_func=warp_func)


def linear(params, x, warp_func=None):
  """Linear function."""
  (linear_noise_variance,) = retrieve_params(
      params, ['linear_noise_variance'], warp_func
  )
  val = nn.Dense(1).apply({'params': linear_noise_variance}, x)
  return get_noise_variance(val, warp_func=warp_func)


def linear_mlp(params, x, warp_func=None):
  """Fully-connected neural net as a noise_variance function."""
  mlp_params, = retrieve_params(params, ['mlp_params'], warp_func)
  val = linear(
      params,
      bf.MLP(params.config['mlp_features']).apply({'params': mlp_params}, x),
      warp_func=warp_func)
  return get_noise_variance(val, warp_func=warp_func)
