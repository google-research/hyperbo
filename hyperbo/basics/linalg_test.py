# coding=utf-8
# Copyright 2026 HyperBO Authors.
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

"""Tests for linalg.py."""
import copy

from absl.testing import absltest
from absl.testing import parameterized
from hyperbo.basics import linalg
import jax
from jax import random
import jax.numpy as jnp
import jax.scipy.linalg as jspla
import numpy as np

grad = jax.grad


def test_grad(fun, params, index, eps=1e-4, cached_cholesky=False):
  key = random.PRNGKey(0)
  _, subkey = random.split(key)
  vec = random.normal(subkey, params[index].shape)
  if index == 0:
    vec = 0.5 * jnp.dot(vec.T, vec)
    unitvec = vec / jnp.sqrt(jnp.vdot(vec, vec))
  else:
    unitvec = vec / jnp.sqrt(jnp.vdot(vec, vec))

  params_copy = copy.deepcopy(params)
  params_copy[index] += eps / 2. * unitvec
  if cached_cholesky:
    params_copy[2] = jspla.cholesky(params_copy[0], lower=True)
  f1 = fun(*params_copy)
  params_copy = copy.deepcopy(params)
  params_copy[index] -= eps / 2. * unitvec
  if cached_cholesky:
    params_copy[2] = jspla.cholesky(params_copy[0], lower=True)
  f2 = fun(*params_copy)
  exact_grad_prod = jnp.vdot(grad(fun, index)(*params), unitvec)

  return {'Numerical': (f1 - f2) / eps, 'Exact': exact_grad_prod}


class LinalgTest(absltest.TestCase):

  def test_inverse_spdmatrix_vector_product(self):
    np.random.seed(1)
    dim = 10
    noise = 1e-3
    num_replicas = 10

    def fun(spd_matrix, x):
      return jnp.dot(x, linalg.inverse_spdmatrix_vector_product(spd_matrix, x))

    def test_grad_at_index(index):
      for _ in range(num_replicas):
        matrix = np.random.randn(dim, dim)
        spd_matrix = matrix.T.dot(matrix) + noise * np.eye(matrix.shape[0])
        x = np.random.randn(dim)
        params = [spd_matrix, x]
        grads = test_grad(fun, params, index)

        numerical_grad = grads['Numerical']
        exact_grad = grads['Exact']
        self.assertTrue(jnp.allclose(numerical_grad, exact_grad, rtol=1))

    test_grad_at_index(0)
    test_grad_at_index(1)

  def test_inverse_spdmatrix_vector_product_cached_cholesky(self):
    """Tests if the gradient works when the Cholesky factor is given."""

    np.random.seed(1)
    dim = 10
    noise = 1e-3
    num_replicas = 10

    def fun(spd_matrix, x, cached_cholesky):
      return jnp.dot(
          x,
          linalg.inverse_spdmatrix_vector_product(
              spd_matrix, x, cached_cholesky=cached_cholesky))

    def test_grad_at_index(index):
      for _ in range(num_replicas):
        matrix = np.random.randn(dim, dim)
        spd_matrix = matrix.T.dot(matrix) + noise * np.eye(matrix.shape[0])
        chol_factor = jspla.cholesky(spd_matrix, lower=True)
        x = np.random.randn(dim)
        params = [spd_matrix, x, chol_factor]
        grads = test_grad(fun, params, index, cached_cholesky=True)

        numerical_grad = grads['Numerical']
        exact_grad = grads['Exact']
        print(numerical_grad, exact_grad)
        self.assertTrue(jnp.allclose(numerical_grad, exact_grad, rtol=1))

    test_grad_at_index(0)
    test_grad_at_index(1)


class MatrixRightHandSideTest(parameterized.TestCase):

  @parameterized.product(
      columns=[1, 3], cached=[False, True], compiled=[False, True])
  def test_matches_independent_vector_solves(self, columns, cached, compiled):
    matrix = jnp.array([[3., 0.5], [0.5, 2.]])
    rhs = jnp.arange(1., 2 * columns + 1).reshape(2, columns)
    cotangent = jnp.linspace(-1., 2., 2 * columns).reshape(2, columns)

    def objective(a, x):
      chol = jspla.cholesky(a, lower=True) if cached else None
      solution = linalg.inverse_spdmatrix_vector_product(a, x, chol)
      return jnp.sum(solution * cotangent)

    def reference(a, x):
      chol = jspla.cholesky(a, lower=True) if cached else None
      return sum(
          jnp.vdot(linalg.inverse_spdmatrix_vector_product(a, x[:, i], chol),
                   cotangent[:, i]) for i in range(columns))

    evaluate = jax.value_and_grad(objective, argnums=(0, 1))
    if compiled:
      evaluate = jax.jit(evaluate)
    actual_value, (actual_matrix, actual_rhs) = evaluate(matrix, rhs)
    expected_value, (expected_matrix, expected_rhs) = jax.value_and_grad(
        reference, argnums=(0, 1))(matrix, rhs)
    self.assertEqual(actual_matrix.shape, matrix.shape)
    self.assertEqual(actual_rhs.shape, rhs.shape)
    np.testing.assert_allclose(
        actual_value, expected_value, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        actual_matrix, expected_matrix, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        actual_rhs, expected_rhs, rtol=1e-6, atol=1e-6)

    # Independently verify derivatives along the symmetric matrix domain.
    direction = jnp.array([[0.3, -0.2], [-0.2, 0.4]])
    eps = 1e-3
    numerical = (objective(matrix + eps * direction, rhs)
                 - objective(matrix - eps * direction, rhs)) / (2 * eps)
    np.testing.assert_allclose(
        jnp.vdot(actual_matrix, direction), numerical, rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(
        actual_rhs, jnp.linalg.solve(matrix, cotangent), rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
  absltest.main()
