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

"""Basis functions."""
from typing import Sequence

from flax import linen as nn
from hyperbo.gp_utils import utils
import jax.numpy as jnp


class MLP(nn.Module):
  """Multi-layer perceptron basis functions.

  Attributes:
    features: Sequence[int] describing output features dimensions.
  """
  features: Sequence[int]

  @nn.compact
  def __call__(self, x):
    for feat in self.features:
      x = nn.tanh(nn.Dense(feat)(x))
    return x


def init_mlp_with_shape(key, params, input_shape):
  """Initialize mlp parameters in params with desired input shape."""
  input_shape = list(input_shape)
  input_shape[0] = 0
  init_val = jnp.ones(input_shape, jnp.float32)
  params.model['mlp_params'] = MLP(params.config['mlp_features']).init(
      key, init_val)['params']


class KumarWarp(nn.Module):
  """Kumaraswamy CDF warping on each dimension of the input."""

  @nn.compact
  def __call__(self, inputs):

    # assert (jnp.all(inputs >= 0)
    #         and jnp.all(inputs <= 1)), 'Inputs must lie in [0, 1]'

    a = self.param('a', nn.initializers.zeros, inputs.shape[-1])
    b = self.param('b', nn.initializers.zeros, inputs.shape[-1])

    a = utils.squareplus_warp(a)
    b = utils.squareplus_warp(b)
    return 1 - (1 - inputs**a)**b


def init_kumar_warp_with_shape(key, params, input_shape):
  """Initialize mlp parameters in params with desired input shape."""
  input_shape = list(input_shape)
  input_shape[0] = 0
  init_val = jnp.ones(input_shape, jnp.float32)
  params.model['kumar_params'] = KumarWarp().init(key, init_val)['params']
