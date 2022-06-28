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

"""Utility functions for handling GP parameters."""

import logging
import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Union

from hyperbo.basics import definitions as defs
import jax
import ml_collections

from tensorflow.io import gfile

GPParams = defs.GPParams


def save_params(filenm: str, params: Union[GPParams, Dict[str, Any]]):
  """Save to file."""
  if not isinstance(params, dict):
    params = params.__dict__
  params = jax.tree_map(lambda x: str(x) if callable(x) else x, params)
  dirnm = os.path.dirname(filenm)
  if not gfile.Exists(dirnm):
    gfile.MakeDirs(dirnm)
  with gfile.GFile(filenm, 'wb') as f:
    pickle.dump(params, f)


def load_params(filenm: str, use_gpparams: bool = True) -> Any:
  """Load from file."""
  if not gfile.Exists(filenm):
    msg = f'{filenm} does not exist.'
    logging.info(msg=msg)
    print(msg)
    return None
  with gfile.GFile(filenm, 'rb') as f:
    params_dict = pickle.load(f)
  if use_gpparams:
    return GPParams(**params_dict)
  else:
    return params_dict


def _verify_params(model_params: Dict[str, Any], expected_keys: List[str]):
  """Verify that dictionary params has the expected keys."""
  if not set(expected_keys).issubset(set(model_params.keys())):
    raise ValueError(f'Expected parameters are {sorted(expected_keys)}, '
                     f'but received {sorted(model_params.keys())}.')


def retrieve_params(
    params: GPParams,
    keys: List[str],
    warp_func: Optional[Dict[str, Callable[[Any], Any]]] = None) -> List[Any]:
  """Returns a list of parameter values (warped if specified) by keys' order."""
  model_params = params.model
  _verify_params(model_params, keys)
  if warp_func:
    values = [
        warp_func[key](model_params[key])
        if key in warp_func else model_params[key] for key in keys
    ]
  else:
    values = [model_params[key] for key in keys]
  return values


def encode_model_filename(config: ml_collections.ConfigDict):
  """Encode a config into a model filename.

  Args:
    config: ml_collections.ConfigDict.

  Returns:
    file name string.
  """
  if config.data_loader_name == 'pd1':
    model_key = '-'.join(
        (config.test_workload, str(config.seed), config.mean_func_name,
         config.cov_func_name, config.init_params.config['method'],
         config.init_params.config['objective'], str(config.num_remove),
         str(config.p_observed), str(config.p_remove)))
  elif 'hpob' in config.data_loader_name:
    model_key = '-'.join(
        (config.search_space_index, str(config.seed), config.mean_func_name,
         config.cov_func_name, config.init_params.config['method']))

    if isinstance(config.init_params.config['mlp_features'], tuple):
      model_key = '-'.join(
          (model_key, str(config.init_params.config['mlp_features'])))
    if config.use_surrogate_train:
      model_key = '-'.join(
          (model_key, 'use_surrogate_train'))
    if config.wild_card_train:
      model_key = '-'.join(
          (model_key, f'wild_card_train={config.wild_card_train}'))
    if config.normalize_y:
      model_key = '-'.join(
          (model_key, 'normalize_y'))
    if config.output_log_warp:
      model_key = '-'.join(
          (model_key, 'output_log_warp'))
  else:
    raise NotImplementedError(
        f'Filename encoder not implemented for {config.data_loader_name}')
  return os.path.join(config.model_dir, model_key + '.pkl')


def log_params_loss(step: int,
                    params: GPParams,
                    loss: float,
                    warp_func: Optional[Dict[str, Callable[[Any], Any]]] = None,
                    params_save_file: Optional[str] = None):
  """Log intermediate information of params and nll during training."""
  model_params = params.model
  keys = list(model_params.keys())
  retrieved_params = dict(
      zip(keys, retrieve_params(params, keys, warp_func=warp_func)))
  if params_save_file is not None:
    save_params(params_save_file, params)
  logging.log(
      msg=f'iter={step}, loss={loss}, '
      f'params.model after warping={retrieved_params}',
      level=logging.INFO)
