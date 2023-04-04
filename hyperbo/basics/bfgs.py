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

"""A wrapper for a Jax BFGS optimizer."""
import logging
from typing import Any, Dict, Tuple
import jax
from jax import flatten_util
import jax.scipy.optimize as jspopt


def bfgs(fun, x0: Dict[str, Any], tol: float,
         max_training_step: int) -> Tuple[Dict[str, Any], jax.Array]:
  """Minimization of scalar function of a pytree input.

  Args:
    fun: the objective function to be minimized, ``fun(x, *args) -> float``,
      where ``x`` is a pytree.
    x0: initial guess.
    tol: tolerance for termination.
    max_training_step: maximum number of iterations to perform.

  Returns:
  x: Dict[str, Any], optimized target.
  results.fun: optimized function value.
  """
  flat_x0, unravel_pytree = flatten_util.ravel_pytree(x0)

  def flat_fun(flat_x):
    x = unravel_pytree(flat_x)
    return fun(x)

  results = jspopt.minimize(
      flat_fun,
      flat_x0,
      method='bfgs',
      tol=tol,
      options={'max_training_step': max_training_step})
  x = unravel_pytree(results.x)
  logging.info(msg=f'BFGS results = {results}')
  return x, results.fun
