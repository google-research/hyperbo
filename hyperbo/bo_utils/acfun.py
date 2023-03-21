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

"""Acquisition functions for Bayesian optimization."""
import functools
from typing import Any, Callable, Union

from hyperbo.gp_utils import gp
import jax.numpy as jnp
import jax.random as jrd
import jax.scipy as jsp

partial = functools.partial


def random_search(model, x_queries, **unused_kwargs):
  """Returns a uniformly sampled random array."""
  assert model.rng is not None, 'Random search requires random key.'
  key, subkey = jrd.split(model.rng)
  model.rng = key
  return jrd.uniform(subkey, (x_queries.shape[0], 1))


def acfun_wrapper(acfun_sub: Callable[[jnp.array, jnp.array, Any], jnp.array],
                  acfun_callback_default: Callable[[gp.GP, Union[int, str]],
                                                   Any]):
  """Wrapper for sub acquisition function.

  Args:
    acfun_sub: sub acquisition functions such as ucb_sub.
    acfun_callback_default: default callback function that returns acquisition
      parameter such as beta for ucb_sub given model.

  Returns:
    Acquisition function.
  """

  def acquisition_function(
      *,
      model: gp.GP,
      sub_dataset_key: Union[int, str],
      x_queries: jnp.array,
      acfun_callback: Callable[..., Any] = acfun_callback_default,
  ):
    """Acquisition function.

    Args:
      model: gp.GP.
      sub_dataset_key: key for the sub-dataset that is being queried in
        model.dataset. sub_dataset_key must be a key of model.dataset. See
        attribute dataset in gp.GP for more details.
      x_queries: n' x d input array to be queried.
      acfun_callback: callback function that returns acquisition parameter such
        as beta for ucb_sub given model.

    Returns:
      n' x 1 dimensional evaluations of acfun_sub on x_queries.
    """
    if isinstance(model, gp.HGP):
      predicts = model.predict(
          x_queries,
          sub_dataset_key=sub_dataset_key,
          full_cov=False,
          with_noise=True)
      acfun_param = acfun_callback(model, sub_dataset_key)
      ac_vals = []
      for mu, var in predicts:
        ac_vals.append(acfun_sub(mu, jnp.sqrt(var), acfun_param))
      ac_val = jnp.mean(ac_vals, axis=0)
    else:
      mu, var = model.predict(
          x_queries,
          sub_dataset_key=sub_dataset_key,
          full_cov=False,
          with_noise=True)
      acfun_param = acfun_callback(model, sub_dataset_key)
      ac_val = acfun_sub(mu, jnp.sqrt(var), acfun_param)
    return ac_val

  return acquisition_function


def expected_improvement_sub(mu, std, target):
  """Sub function to compute the expected improvement acquisition function.

  Args:
    mu: n x 1 posterior mean of n query points.
    std: n x 1 posterior standard deviation of n query points (same order as
      mu).
    target: target value to be improved over.

  Returns:
    Expected improvement.
  """
  gamma = (target - mu) / std
  return (jsp.stats.norm.pdf(gamma) - gamma *
          (1 - jsp.stats.norm.cdf(gamma))) * std


def probability_of_improvement_sub(mu, std, target):
  """Sub function to compute the probability of improvement acquisition function.

  Args:
    mu: n x 1 posterior mean of n query points.
    std: n x 1 posterior standard deviation of n query points (same order as
      mu).
    target: target value to be improved over.

  Returns:
    Negative of target z-score as an equivalence of probability of improvement.
  """
  gamma = (target - mu) / std
  return -gamma


def ucb_sub(mu, std, beta=3.):
  """Sub function to compute the upper confidence bound acquisition function.

  Args:
    mu: n x 1 posterior mean of n query points.
    std: n x 1 posterior standard deviation of n query points (same order as
      mu).
    beta: scaling parameter between mean and standard deviation in UCB
      acquisition function. Default value is 3.

  Returns:
    Upper confidence bound.
  """
  return mu + beta * std


def ei_callback_default(model, key, **unused_kwargs):
  if key not in model.dataset or model.dataset[key].y.shape[0] == 0:
    return 0.0
  return jnp.max(model.dataset[key].y)


expected_improvement = acfun_wrapper(
    acfun_sub=expected_improvement_sub,
    acfun_callback_default=ei_callback_default,
)

ei = expected_improvement


def pi_callback_default(model, key, zeta=0.1, use_std=False, **unused_kwargs):
  if key not in model.dataset or model.dataset[key].y.shape[0] == 0:
    return 0.0
  if use_std:
    return jnp.max(model.dataset[key].y) + zeta * jnp.std(model.dataset[key].y)
  else:
    return jnp.max(model.dataset[key].y) + zeta


probability_of_improvement = acfun_wrapper(
    acfun_sub=probability_of_improvement_sub,
    acfun_callback_default=pi_callback_default)

pi = probability_of_improvement
pi2 = acfun_wrapper(
    acfun_sub=probability_of_improvement_sub,
    acfun_callback_default=partial(pi_callback_default, use_std=True))
pi3 = acfun_wrapper(
    acfun_sub=probability_of_improvement_sub,
    acfun_callback_default=partial(pi_callback_default, zeta=0.05))

# UCB variants with 4., 3. or 2. as the UCB coefficient.
# Default UCB has coefficient 3.
ucb4 = acfun_wrapper(acfun_sub=ucb_sub, acfun_callback_default=lambda a, b: 4.)
ucb3 = acfun_wrapper(acfun_sub=ucb_sub, acfun_callback_default=lambda a, b: 3.)
ucb2 = acfun_wrapper(acfun_sub=ucb_sub, acfun_callback_default=lambda a, b: 2.)
ucb = ucb3

rand = random_search
