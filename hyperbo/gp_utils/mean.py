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

"""Mean function library for GP."""
import functools

from flax import linen as nn
from hyperbo.basics import params_utils
from hyperbo.gp_utils import basis_functions as bf
import jax
import jax.numpy as jnp

vmap = jax.vmap

retrieve_params = params_utils.retrieve_params


def mean_vector(mean_func):
  """Decorator to mean functions to obtain the mean vector."""

  @functools.wraps(mean_func)
  def vector_map(params, vx, warp_func=None):
    """Returns the mean vector of input array vx.

    Args:
      params: parameters for the mean function.
      vx: n x d dimensional input array representing n data points.
      warp_func: optional dictionary that specifies the warping function for
        each parameter.

    Returns:
      The n dimensional mean vector derived from mean function evaluations
        on every input from vx.
    """
    # pylint: disable=unnecessary-lambda
    parameterized_mean_func = lambda x: mean_func(params, x, warp_func)
    return vmap(parameterized_mean_func)(vx)

  return vector_map


@mean_vector
def zero(*_, **_kwargs):
  """Zero mean function."""
  return jnp.full((1,), 0)


@mean_vector
def constant(params, _, warp_func=None):
  """Constant mean function."""
  val, = retrieve_params(params, ['constant'], warp_func)
  return jnp.full((1,), val)


def linear(params, x, warp_func=None):
  """Linear function."""
  linear_mean, = retrieve_params(params, ['linear_mean'], warp_func)
  return nn.Dense(1).apply({'params': linear_mean}, x)


def linear_mlp(params, x, warp_func=None):
  """Fully-connected neural net as a mean function."""
  mlp_params, = retrieve_params(params, ['mlp_params'], warp_func)
  return linear(
      params,
      bf.MLP(params.config['mlp_features']).apply({'params': mlp_params}, x),
      warp_func=warp_func)
