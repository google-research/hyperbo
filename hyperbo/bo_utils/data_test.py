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

"""Test data.py.

Use the following to debug nan.

from jax import config
config.update('jax_debug_nans', True)
"""
import logging

from absl.testing import absltest
from absl.testing import parameterized
from hyperbo.basics import data_utils
from hyperbo.basics import definitions as defs
from hyperbo.bo_utils import const
from hyperbo.bo_utils import data
from hyperbo.gp_utils import kernel
from hyperbo.gp_utils import mean
import jax
import jax.numpy as jnp
import numpy as np

GPParams = defs.GPParams
ALL_DATASETS = const.HYPERBO_DATASETS


class DataTest(parameterized.TestCase):
  """Tests for data.py."""

  @parameterized.named_parameters(ALL_DATASETS.items())
  def test_dataset_shape(self, dataset_creator):
    """Test that values returned from functions in acfun has the right shape."""
    if dataset_creator.__name__ == 'random':
      key = jax.random.PRNGKey(0)
      n_observed, n_queries = 20, 10
      n_func_historical, m_points_historical = 3, 7
      dim = 5
      params = GPParams(
          model={
              'constant': 5.,
              'lengthscale': 1.,
              'signal_variance': 1.0,
              'noise_variance': 0.01,
          })
      mean_func = mean.constant
      cov_func = kernel.squared_exponential
      dataset, sub_dataset_key, queried_subdataset = dataset_creator(
          key=key,
          mean_func=mean_func,
          cov_func=cov_func,
          params=params,
          dim=dim,
          n_observed=n_observed,
          n_queries=n_queries,
          n_func_historical=n_func_historical,
          m_points_historical=m_points_historical)
      self.assertLen(dataset, n_func_historical + 1)
      self.assertEqual(sub_dataset_key in dataset, True)
      for dataset_key in dataset.keys():
        if dataset_key == sub_dataset_key:
          self.assertEqual(dataset[dataset_key].x.shape, (n_observed, dim))
          self.assertEqual(dataset[dataset_key].y.shape, (n_observed, 1))
        else:
          self.assertEqual(dataset[dataset_key].x.shape,
                           (m_points_historical, dim))
          self.assertEqual(dataset[dataset_key].y.shape,
                           (m_points_historical, 1))
      self.assertEqual(queried_subdataset.x.shape, (n_queries, dim))
      self.assertEqual(queried_subdataset.y.shape, (n_queries, 1))

  @parameterized.named_parameters(
      {
          'testcase_name': 'three_duplicates',
          'x': [[2., 3.], [2., 1.], [2., 3], [2., 3.]],
          'y': [[1.], [2.], [4.], [3]],
          # Arrays are sorted before deduplication.
          'expected_x': [[2., 1.], [2., 3.]],
          'expected_y': [[2.], [4.]],
      },
      {
          'testcase_name': 'no_duplicates',
          'x': [[1., 2.], [3., 4.]],
          'y': [[1.], [2.]],
          'expected_x': [[1., 2.], [3., 4.]],
          'expected_y': [[1.], [2.]],
      })
  def test_deduplicate(self, x, y, expected_x, expected_y):
    actual_x, actual_y = data._deduplicate(
        np.array(x), np.array(y), dataset_name='')
    np.testing.assert_array_equal(actual_x, expected_x)
    np.testing.assert_array_equal(actual_y, expected_y)

  @parameterized.parameters(
      (True,),
      (False,),
  )
  def test_normalize_maf_dataset(self, neg_error_to_accuracy):
    # max_values = [3, 2, 2]
    # min_values [-1, 1, 1]
    maf_dataset = {
        'workload_a': {
            'X': np.array([[-1, 2, 1], [2, 2, 2]]),
            'Y': np.array([[-.1], [0.]])
        },
        'workload_b': {
            'X': np.array([[1, 1, 2], [3, 2, 2]]),
            'Y': np.array([[-.9], [-.2]])
        }
    }

    def update_y(y):
      if neg_error_to_accuracy:
        return y + 1
      else:
        return y

    expected_normalized_dataset = {
        'workload_a': {
            'X': np.array([[0, 1, 0], [.75, 1, 1]]),
            'Y': update_y(np.array([[-.1], [0.]]))
        },
        'workload_b': {
            'X': np.array([[.5, 0, 1], [1, 1, 1]]),
            'Y': update_y(np.array([[-.9], [-.2]]))
        }
    }
    actual_normalized_dataset = data._normalize_maf_dataset(
        maf_dataset, num_hparams=3, neg_error_to_accuracy=neg_error_to_accuracy)
    for wl in maf_dataset:
      np.testing.assert_array_equal(expected_normalized_dataset[wl]['X'],
                                    actual_normalized_dataset[wl]['X'])
      np.testing.assert_array_equal(expected_normalized_dataset[wl]['Y'],
                                    actual_normalized_dataset[wl]['Y'])


if __name__ == '__main__':
  absltest.main()
