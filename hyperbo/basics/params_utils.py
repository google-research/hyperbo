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

"""Utility functions for handling GP parameters."""

import logging
import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Union

from hyperbo.basics import definitions as defs
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np

from tensorflow.io import gfile

GPParams = defs.GPParams
FINAL_PARAM_FILE_INFO = 'FINAL'


def to_list_or_float(x):
  """Transform np.ndarrray or np.float to python list and float, if any."""
  if isinstance(x, jnp.ndarray) or isinstance(x, np.ndarray):
    return x.tolist()
  elif isinstance(x, np.float32) or isinstance(x, np.float64):
    return float(x)
  else:
    return x


def save_to_file(filenm: str, state: Any = None):
  """Save to file."""
  if not state:
    return
  dirnm = os.path.dirname(filenm)
  if not gfile.Exists(dirnm):
    gfile.MakeDirs(dirnm)
  with gfile.GFile(filenm, 'wb') as f:
    pickle.dump(state, f)


def load_from_file(filenm: str):
  if not gfile.Exists(filenm):
    raise FileNotFoundError(f'{filenm} does not exist.')
  with gfile.GFile(filenm, 'rb') as f:
    state = pickle.load(f)
  return state


def save_params(filenm: str,
                params: Union[GPParams, Dict[str, Any]],
                state: Any = None):
  """Save to file."""
  if not isinstance(params, dict):
    params = params.__dict__
  params = jax.tree_map(lambda x: str(x) if callable(x) else x, params)
  if state:
    state = jax.tree_map(lambda x: str(x) if callable(x) else x, state)
  save_to_file(filenm, (params, state))


def load_params(filenm: str,
                use_gpparams: bool = True,
                include_state: bool = False):
  """Load from file."""
  params_dict, state = load_from_file(filenm)
  if use_gpparams:
    params = GPParams(**params_dict)
  else:
    params = params_dict
  if include_state:
    return params, state
  return params


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
  model_key = ''
  if config.data_loader_name == 'pd1':
    model_key = '-'.join(
        (config.test_dataset_index, str(config.seed), config.mean_func_name,
         config.cov_func_name, str(config.init_params.config['mlp_features']),
         config.init_params.config['objective'],
         config.init_params.config['method'],
         str(config.init_params.config['max_training_step']),
         str(config.init_params.config['batch_size']), str(config.num_remove),
         str(config.p_observed), str(config.p_remove)))
    if 'num_irrelevant' in config and config.num_irrelevant:
      model_key = '-'.join((model_key, config.num_irrelevant))
  elif 'hpob' in config.data_loader_name:
    model_key = '-'.join(
        (config.search_space_index, str(config.seed), config.mean_func_name,
         config.cov_func_name, config.init_params.config['method']))

    if isinstance(config.init_params.config['mlp_features'], tuple):
      model_key = '-'.join(
          (model_key, str(config.init_params.config['mlp_features'])))
    if config.use_surrogate_train:
      model_key = '-'.join((model_key, 'use_surrogate_train'))
    if config.wild_card_train:
      model_key = '-'.join(
          (model_key, f'wild_card_train={config.wild_card_train}'))
    if config.normalize_y:
      model_key = '-'.join((model_key, 'normalize_y'))
    if config.output_log_warp:
      model_key = '-'.join((model_key, 'output_log_warp'))
  else:
    raise NotImplementedError(
        f'Filename encoder not implemented for {config.data_loader_name}')

  def get_path(additional_info=FINAL_PARAM_FILE_INFO, model_key_only=False):
    """Generate the path to save model parameters.

    Args:
      additional_info: a string or int that will be appended at the end of the
        filename to encode additional information.
      model_key_only: only return model key if True; otherwise full path.

    Returns:
      full path with filename.
    """
    if model_key_only:
      return model_key
    if not isinstance(config.model_dir, str):
      raise ValueError(f'config.model_dir={config.model_dir} is not valid.')
    if not isinstance(additional_info, str):
      additional_info = str(additional_info)
    if config.method == 'stbo':
      model_spec = '-'.join((
          model_key,
          config.ac_func_name,
          config.method,
          config.test_dataset_index,
          config.test_seed,
      ))
    else:
      model_spec = model_key
    if config.data_loader_name == 'pd1':
      return os.path.join(config.model_dir, model_spec,
                          f'{additional_info}.pkl')
    elif 'hpob' in config.data_loader_name:
      model_file_name = '-'.join((model_spec, additional_info))
      return os.path.join(config.model_dir, model_file_name + '.pkl')

  return get_path


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
  logging.info(msg=f'logging iter={step}, loss={loss}, '
               f'params.model after warping={retrieved_params}')
  if params_save_file is not None:
    logging.info(msg=f'Saving params to {params_save_file}.')
    save_params(params_save_file, params, state=(step, loss))
