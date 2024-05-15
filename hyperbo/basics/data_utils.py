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

"""Utils for data."""
import functools

from absl import logging
from hyperbo.basics import definitions as defs
import jax
import jax.numpy as jnp

partial = functools.partial

SubDataset = defs.SubDataset


def log_dataset(dataset):
  """Log basic facts about dataset."""

  def safe(f):

    def safef(x):
      if not isinstance(x, jnp.ndarray):
        return x
      elif x.shape[0] == 0:
        return jnp.nan
      else:
        return f(x)

    return safef

  logging.info(msg=f'dataset len = {len(dataset)}.')
  logging.info(msg=f'dataset shape: {jax.tree.map(safe(jnp.shape), dataset)}')
  logging.info(
      msg=(
          'dataset mean: '
          f'{jax.tree.map(safe(partial(jnp.mean, axis=0)), dataset)}'
      )
  )
  logging.info(
      msg=(
          'dataset median: '
          f'{jax.tree.map(safe(partial(jnp.median, axis=0)), dataset)}'
      )
  )
  logging.info(
      msg=(
          'dataset min: '
          f'{jax.tree.map(safe(partial(jnp.min, axis=0)), dataset)}'
      )
  )
  logging.info(
      msg=(
          'dataset max: '
          f'{jax.tree.map(safe(partial(jnp.max, axis=0)), dataset)}'
      )
  )


def sub_sample_dataset_iterator(key, dataset, batch_size):
  """Iterator for subsample a dataset such that each sub_dataset has at most batch_size data points.

  Args:
    key: Jax random state.
    dataset: dict of SubDataset.
    batch_size: int, maximum number of data points per sub dataset in a batch.

  Yields:
    A sub sampled dataset batch.
  """
  while True:
    sub_sampled_dataset = {}
    for i, (sub_dataset_key, sub_dataset) in enumerate(dataset.items()):
      if sub_dataset.x.shape[0] >= batch_size:
        key, subkey = jax.random.split(key, 2)
        indices = jax.random.permutation(subkey, sub_dataset.x.shape[0])
        new_sub_dataset = SubDataset(
            x=sub_dataset.x[indices[:batch_size], :],
            y=sub_dataset.y[indices[:batch_size], :],
            aligned=sub_dataset.aligned)
      else:
        new_sub_dataset = sub_dataset
      if isinstance(new_sub_dataset.aligned, str):
        # We have to do this because str is not a Jax supported type.
        new_sub_dataset = SubDataset(
            x=new_sub_dataset.x, y=new_sub_dataset.y, aligned=i)
      sub_sampled_dataset[sub_dataset_key] = new_sub_dataset
    yield sub_sampled_dataset
