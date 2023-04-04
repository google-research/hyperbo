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

"""Convex gradient-based optimization algorithms.

This file contains simple implementations of some convex gradient based
optimization algorithms, specifically for use with non-jittable large scale
functions.

Author: Jasper Snoek.
"""

from absl import logging
from flax.core import frozen_dict
import jax
import jax.numpy as jnp


@jax.jit
def _dict_tensordot(a, b, axes):
  fn = lambda a, b: jnp.tensordot(a, b, axes)
  return jax.tree_map(fn, a, b)


@jax.jit
def _dict_vdot(a, b):
  return jax.tree_util.tree_reduce(jnp.add, jax.tree_map(jnp.vdot, a, b))


def _return_index(fn, index):

  def wrapper(*args, **kwargs):
    result = fn(*args, **kwargs)
    return result[index]

  return wrapper


def backtracking_linesearch(val_and_grad_fn,
                            cur_val,
                            params,
                            grads,
                            direction,
                            alpha=1.,
                            c1=1e-4,
                            c2=0.9,
                            tau=0.5,
                            max_steps=50,
                            has_aux=False,
                            args=tuple()):
  """A simple two-directional backtracking line-search.

  Uses the Armijo–Goldstein and Wolfe conditions to determine the step size.
  These two are generally bundled into what's called the Wolfe conditions. They
  measure whether "sufficient progress" was made given the current step size.
  The Armijo-Goldstein in 0th order (is the value significantly better adjusted
  for the scale of the function) and the Wolfe in 1st order (e.g. is the
  gradient still steep).  The second Wolfe condition requires a gradient eval
  and is generally required to guarantee convergence of a variety of approximate
  second order methods (such as LBFGS).
  This assumes one is minimizing the function fn.

  Args:
    val_and_grad_fn:  The function that is being minimized, of the form fn(x) =
      y.
    cur_val: The current function value, i.e. conditioned on params.
    params: A dict of numpy arrays of parameters passed to fn.
    grads: Gradients of the function fn at position defined by params.
    direction: A dict with directions to take steps in for each of the values,
      corresponding to params.
    alpha: initial step size.
    c1: A scalar search control parameter determining the strength of
      convergence in (0, 1) for the Armijo condition.
    c2: A scalar search control parameter determining the strength of
      convergence in (0, 1) for the curvature confition.
    tau: A scalar search control parameter determining the strength of
      convergence in (0, 1).
    max_steps: Maximum number of times to evaluate fn and take linesearch steps.
    has_aux: Boolean indicating whether fn returns anything in addition to a
      scalar value, as in jax.value_and_grad.
    args: A tuple containing any additional positional arguments to fn, such
      that fn will be called as fn(params, *args)

  Returns:
    new_val: The resulting value achieved by following the linesearch.
    alpha: The determined step size.
  """
  grads_dot_dir = _dict_vdot(grads, direction)
  if grads_dot_dir > 0.:
    logging.info("Incorrect descent direction %f.  Exiting linesearch",
                 grads_dot_dir)
    return params, alpha

  t = c1 * grads_dot_dir
  armijo_cond = lambda x, a: jnp.greater_equal(cur_val + a * t, x)

  def wolfe_curvature_cond(new_grads):
    return jnp.greater_equal(
        _dict_vdot(new_grads, direction), c2 * grads_dot_dir)

  for i in range(max_steps):
    fn = lambda a, b: a + b * alpha
    new_params = jax.tree_map(fn, params, direction)

    new_val, new_grads = val_and_grad_fn(new_params, *args)
    if has_aux:
      new_val = new_val[0]

    logging.info(
        "Linesearch: step %i orig: %f new: %f step size: %f Armijo cond %d", i,
        cur_val, new_val, alpha, armijo_cond(new_val, alpha))

    if jnp.isfinite(new_val) and armijo_cond(new_val, alpha):
      if wolfe_curvature_cond(new_grads):
        logging.info("Satisfied linesearch Wolfe conditions: step %i %f", i,
                     new_val)
        return new_val, alpha
      else:
        alpha *= 2.1

    else:
      alpha *= tau

  if (not jnp.isnan(new_val)) and jnp.isfinite(new_val):
    return new_val, alpha
  else:  # If we hit nans or infs return where we started.
    return cur_val, 0.


@jax.jit
def lbfgs_descent_dir_nocedal(grads, s, y):
  """Compute the descent direction for L-BFGS.

    This computes a very coarse but memory efficient estimate of the
    Hessian-gradient product to determine a linesearch direction.
    This follows the recursive algorithm specified in "Updating Quasi-Newton
    Matrices with Limited Storage", Nocedal '80, p 779.  Note variable names
    mirror those from Nocedal.

  Args:
    grads: A dict where the values are arrays corresponding to the gradients of
      the function being optimized.
    s: A list of dicts of length M containing the difference in gradients
      (corresponding to grads) from the last M LBFGS updates.
    y: A list of dicts of length M containing the difference in parameters
      (corresponding to grads) from the last M LBFGS updates.

  Returns:
    direction: A dict corresponding to descent directions in similar form to
      grads.
  """

  bound = len(s)
  q = jax.tree_map(lambda x: -x, grads)
  inv_p = [1. / _dict_vdot(y[i], s_i) for i, s_i in enumerate(s)]
  alphas = {}
  for i in range(bound - 1, -1, -1):
    alpha = inv_p[i] * _dict_vdot(s[i], q)
    alphas[i] = alpha
    q = jax.tree_map(lambda a, b, alpha=alpha: a - alpha * b, q, y[i])

  gamma_k = _dict_vdot(s[-1], y[-1]) / _dict_vdot(y[-1], y[-1])
  direction = jax.tree_map(lambda x: gamma_k * x, q)

  for i in range(0, bound):
    beta = inv_p[i] * _dict_vdot(y[i], direction)
    step = (alphas[i] - beta)
    fn = lambda a, b, step=step: a + b * step
    direction = jax.tree_map(fn, direction, s[i])

  return direction


def lbfgs(fn,
          params,
          memory=10,
          ls_steps=50,
          steps=100,
          alpha=1.,
          tol=1e-6,
          ls_tau=0.5,
          args=tuple(),
          has_aux=False,
          val_and_grad_fn=None,
          state=None,
          callback=None):
  """Optimize a function with the lbfgs algorithm.

    This implementation allows for dictionaries of parameters and the
      possibility that the function fn can not be jitted (e.g. contains a pmap).
      Thus it makes use of native python loops but can be jitted externally
      to make the optimization loops faster.

  Args:
    fn: The function to be minimized, called with a single argument params.
    params: A dict of parameters to be passed to the function fn.  The values
      must be dict or numpy arrays.
    memory: The number of steps of history to store for the algorithm.  This
      governs the accuracy of the underlying Hessian approximation while trading
      off the memory usage of the algorithm.
    ls_steps: Number of linesearch steps to do at each LBFGS iteration.
    steps: The total number of optimization steps to perform.
    alpha: Initial step size for the linesearch.
    tol: Convergence tolerance.
    ls_tau: Scalar to multiply the step size by for each linesearch increment,
      in (0, 1)
    args: A tuple containing additional positional arguments to pass to fn, as
      in result = fn(params, *args)
    has_aux: Boolean indicating whether fn returns anything in addition to a
      scalar value, as in jax.value_and_grad.
    val_and_grad_fn: A function that returns the value and gradient of fn, as
      provided by jax.value_and_grad.
    state: A list or tuple containing internal state of the optimizer, to be
      passed in if this is called multiple times in a row to maintain the
      Hessian estimate.
    callback: an optional callback function.

  Returns:
    params: A new set of parameters corresponding to the result of the
      optimization.
    state: A tuple containing the state of the optimizer, i.e. this is to be
      passed back in to the function to reconstruct the Hessian estimate if this
      is called repeatedly.
  """
  if val_and_grad_fn is None:
    val_and_grad_fn = jax.value_and_grad(fn, has_aux=has_aux)

  if isinstance(params, frozen_dict.FrozenDict):
    # Flax generates parameters as FrozenDict, whose copy() function
    # takes a dict argument as input.
    copy_fn = lambda x: x.copy({})
  else:
    copy_fn = lambda x: x.copy()

  if state is None:
    s_k = []
    y_k = []

    val, grads = val_and_grad_fn(params, *args)
    if callback is not None:
      callback(step=0, model_params=params, loss=val)

    grad_norm = _dict_vdot(grads, grads)
    if grad_norm <= tol:
      logging.info("LBFGS converged at start.")
      return val, params, None

    if has_aux:  # Grab just the loss if fn returns multiple things.
      val, aux = val

    descent_dir = jax.tree_map(lambda x: -x, grads)
    old_params = copy_fn(params)
    old_grads = copy_fn(grads)
    init_alpha = 1. / jnp.sqrt(grad_norm)
    new_val, step_size = backtracking_linesearch(
        val_and_grad_fn,
        val,
        params,
        grads,
        descent_dir,
        init_alpha,
        tau=ls_tau,
        args=args,
        has_aux=has_aux,
        max_steps=ls_steps)
    if new_val < val:
      params = jax.tree_map(lambda a, b: a + b * step_size, params,
                                 descent_dir)
    else:
      logging.info("Linesearch did not make progress.")
      new_val = (new_val, aux) if has_aux else new_val
      return new_val, params, (s_k, y_k, old_grads, old_params)
  else:
    s_k, y_k, old_grads, old_params = state

  for i in range(1, steps+1):
    val, grads = val_and_grad_fn(params, *args)
    if has_aux:
      val, aux = val

    # Convergence check.  If Frobenius norm of gradients is less than tol, stop.
    grad_norm = _dict_vdot(grads, grads)
    if grad_norm <= tol:
      logging.info("LBFGS converged in %d steps", i)
      new_val = val
      break

    if old_grads is not None:
      fn = lambda a, b, c: -a + b - c
      if len(s_k) > memory:
        y_k[0] = jax.tree_map(fn, y_k[0], grads, old_grads)
        s_k[0] = jax.tree_map(fn, s_k[0], params, old_params)
        y_k.append(y_k[0])
        s_k.append(s_k[0])
      else:
        y_k.append(jax.tree_map(jnp.subtract, grads, old_grads))
        s_k.append(jax.tree_map(jnp.subtract, params, old_params))

    if len(s_k) > memory:
      s_k = s_k[-memory:]
      y_k = y_k[-memory:]
    old_params = copy_fn(params)
    old_grads = copy_fn(grads)

    magnitude = _dict_vdot(y_k[-1], s_k[-1])
    logging.info("LBFGS step %d val: %f", i, val)
    if callback is not None:
      callback(step=i, model_params=params, loss=val)

    if jnp.isfinite(magnitude) and magnitude >= tol:
      descent_dir = lbfgs_descent_dir_nocedal(grads, s_k, y_k)
      new_val, step_size = backtracking_linesearch(
          val_and_grad_fn,
          val,
          params,
          grads,
          descent_dir,
          alpha,
          args=args,
          tau=ls_tau,
          has_aux=has_aux,
          max_steps=ls_steps)
      if new_val >= val:
        logging.info("Linesearch did not make progress.")
        break

      params = jax.tree_map(lambda a, b: a + b * step_size, params,
                                 descent_dir)
    else:
      new_val = val
      logging.info("LBFGS terminating due to instability.")
      break

  if has_aux:
    return (new_val, aux), params, (s_k, y_k, old_grads, old_params)
  else:
    return new_val, params, (s_k, y_k, old_grads, old_params)
