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

"""Independent GP samples should contribute additive likelihoods and gradients."""

from absl.testing import absltest
from absl.testing import parameterized
from hyperbo.basics import definitions as defs
from hyperbo.gp_utils import kernel
from hyperbo.gp_utils import mean
from hyperbo.gp_utils import objectives
import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

X = jnp.array([[0.0], [0.4], [1.2]])
Y = jnp.array([[0.2, -1.0, 2.0], [1.3, 0.7, -0.4], [-0.5, 0.4, 1.1]])


def params(signal=1.2):
  return defs.GPParams(
      model={
          "constant": 0.3,
          "lengthscale": 0.7,
          "signal_variance": signal,
          "noise_variance": 0.2,
      }
  )


def nll(y, cholesky=True, par=None, **kwargs):
  return objectives.neg_log_marginal_likelihood(
      mean.constant,
      kernel.squared_exponential,
      par or params(),
      {"observations": defs.SubDataset(x=X, y=y, aligned="shared")},
      exclude_aligned=False,
      use_cholesky=cholesky,
      **kwargs,
  )


class MultisampleLikelihoodTest(parameterized.TestCase):

  @parameterized.product(cholesky=[True, False], columns=[1, 2, 3])
  def test_shared_covariance_matches_independent_multivariate_normals(
      self, cholesky, columns
  ):
    y = Y[:, :columns]
    x = np.asarray(X, dtype=np.float64)
    covariance = 1.2 * np.exp(-0.5 * ((x - x.T) / 0.7) ** 2) + np.eye(3) * (
        0.2 + 1e-6
    )
    expected = -sum(
        stats.multivariate_normal.logpdf(
            np.asarray(y[:, i], dtype=np.float64),
            mean=np.full(3, 0.3),
            cov=covariance,
        )
        for i in range(columns)
    )
    actual, by_key = nll(y, cholesky, return_key2nll=True)
    np.testing.assert_allclose(actual, expected, rtol=2e-6)
    np.testing.assert_allclose(by_key["observations"], expected, rtol=2e-6)

  @parameterized.parameters(True, False)
  def test_replicating_a_sample_scales_the_likelihood_linearly(self, cholesky):
    single = nll(Y[:, :1], cholesky)
    repeated = nll(jnp.repeat(Y[:, :1], 3, axis=1), cholesky)
    np.testing.assert_allclose(repeated, 3 * single, rtol=2e-6)

  @parameterized.parameters(True, False)
  def test_hyperparameter_gradient_is_the_sum_of_independent_gradients(
      self, cholesky
  ):
    objective = lambda signal: nll(Y, cholesky, params(signal))
    independent = lambda signal: sum(
        nll(Y[:, i : i + 1], cholesky, params(signal)) for i in range(3)
    )
    actual = jax.value_and_grad(objective)(jnp.array(1.2))
    expected = jax.value_and_grad(independent)(jnp.array(1.2))
    for a, b in zip(actual, expected):
      np.testing.assert_allclose(a, b, rtol=2e-5, atol=1e-6)

  def test_cholesky_objective_and_gradient_work_under_jit(self):
    objective = lambda signal: nll(Y, True, params(signal))
    value, gradient = jax.jit(jax.value_and_grad(objective))(jnp.array(1.2))
    independent = lambda signal: sum(
        nll(Y[:, i : i + 1], True, params(signal)) for i in range(3)
    )
    expected = jax.value_and_grad(independent)(jnp.array(1.2))
    np.testing.assert_allclose(value, expected[0], rtol=2e-6)
    np.testing.assert_allclose(gradient, expected[1], rtol=2e-5, atol=1e-6)

  def test_default_exclusion_and_empty_data_are_unchanged(self):
    data = {"a": defs.SubDataset(x=X, y=Y, aligned="shared")}
    self.assertEqual(
        objectives.neg_log_marginal_likelihood(
            mean.constant, kernel.squared_exponential, params(), data
        ),
        0.0,
    )
    self.assertEqual(
        objectives.neg_log_marginal_likelihood(
            mean.constant, kernel.squared_exponential, params(), {}
        ),
        0.0,
    )


if __name__ == "__main__":
  absltest.main()
