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


def load_params(filenm: str,
                use_gpparams: bool = True) -> Any:
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


def retrieve_params(params: GPParams,
                    keys: List[str],
                    warp_func: Optional[Dict[str, Callable[[Any], Any]]] = None,
                    is_gpparams: Optional[bool] = True) -> List[Any]:
  """Returns a list of parameter values (warped if specified) by keys' order."""
  if is_gpparams:
    params = params.model
  _verify_params(params, keys)
  if warp_func:
    values = [
        warp_func[key](params[key]) if key in warp_func else params[key]
        for key in keys
    ]
  else:
    values = [params[key] for key in keys]
  return values
