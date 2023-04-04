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

"""Prior distribution for model parameters."""

import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


def _square_plus(x):
  return (x + jnp.sqrt(x**2 + 4)) / 2


def kumar_prior(params):
  dist = tfd.TruncatedNormal(0., 1., -2., 2.)
  prior_ll = lambda x: dist.log_prob(jnp.log(_square_plus(x)))
  return jnp.sum(jnp.array([prior_ll(v) for v in params.values()]))


noise_prior = lambda x: jnp.sum(tfd.Normal(0., 0.1).log_prob(x))
lognormal_prior = lambda x: jnp.sum(tfd.LogNormal(0., 1.).log_prob(x))
constant_prior = lambda x: jnp.sum(tfd.Normal(0., 1.).log_prob(x))
horseshoe_prior = lambda x, tau: jnp.log(jnp.log(1. + 3. * (tau / x)**2))

DEFAULT_PRIORS = {
    # 'kumar_params': kumar_prior,
    'noise_variance': noise_prior,
    'signal_variance': lognormal_prior,
    # 'lengthscale': lognormal_prior,
    'constant': constant_prior,
}
