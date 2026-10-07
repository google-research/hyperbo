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

"""Tail accuracy regressions for expected improvement."""

from absl.testing import absltest
from absl.testing import parameterized
from hyperbo.basics import definitions
from hyperbo.bo_utils import acfun
from hyperbo.gp_utils import gp
from hyperbo.gp_utils import kernel
from hyperbo.gp_utils import mean
import jax
import jax.numpy as jnp
import numpy as np
from scipy import integrate
from scipy import stats


def tail_expectation(gamma):
  # Integrate the positive improvement directly, independently of CDF subtraction.
  return integrate.quad(
      lambda excess: excess * stats.norm.pdf(gamma + excess),
      0.0,
      np.inf,
      epsabs=1e-30,
      epsrel=1e-10,
  )[0]


class ExpectedImprovementTailTest(parameterized.TestCase):

  @parameterized.parameters(-5.0, 0.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0)
  def test_values_match_positive_improvement_integral(self, gamma):
    std = jnp.array([[0.5], [1.0], [2.0]], dtype=jnp.float32)
    target = 10.0
    mu = target - gamma * std
    expected = np.asarray(std) * tail_expectation(gamma)
    for fn in (
        acfun.expected_improvement_sub,
        jax.jit(acfun.expected_improvement_sub),
    ):
      actual = fn(mu, std, target)
      np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=0)
      self.assertTrue(bool(jnp.all(actual >= 0)))
      self.assertEqual(actual.shape, (3, 1))

  @parameterized.parameters(0.0, 2.0, 5.0, 6.0, 8.0)
  def test_mean_and_scale_gradients_match_gaussian_identities(self, gamma):
    def f(mu, std):
      return acfun.expected_improvement_sub(mu, std, jnp.array(gamma))

    for derivative in (jax.grad(f, (0, 1)), jax.jit(jax.grad(f, (0, 1)))):
      dmu, dstd = derivative(jnp.array(0.0), jnp.array(1.0))
      np.testing.assert_allclose(dmu, stats.norm.sf(gamma), rtol=7e-4, atol=0)
      np.testing.assert_allclose(dstd, stats.norm.pdf(gamma), rtol=7e-4, atol=0)

  def test_public_gp_acquisition_uses_the_correct_tail(self):
    model = gp.GP(
        dataset={},
        mean_func=mean.constant,
        cov_func=kernel.squared_exponential,
        params=definitions.GPParams(
            model={
                "constant": 0.0,
                "lengthscale": 1.0,
                "signal_variance": 1.0,
                "noise_variance": 0.0,
            }
        ),
    )
    actual = acfun.expected_improvement(
        model=model,
        sub_dataset_key="new",
        x_queries=jnp.ones((3, 2)),
        acfun_callback=lambda unused_model, unused_key: 5.0,
    )
    np.testing.assert_allclose(actual, tail_expectation(5.0), rtol=5e-4, atol=0)


if __name__ == "__main__":
  absltest.main()
