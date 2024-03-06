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

"""Bayesian optimization (BO) loop for sequential queries."""
import dataclasses
import time
from typing import Callable, Optional, Sequence, Tuple, Union, Any

from absl import logging
from hyperbo.basics import definitions as defs
from hyperbo.bo_utils import const
from hyperbo.gp_utils import gp
from hyperbo.gp_utils import objectives as obj
from hyperbo.gp_utils import priors
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np

SubDataset = defs.SubDataset
INPUT_SAMPLERS = const.INPUT_SAMPLERS


def get_best_datapoint(sub_dataset):
  """Return the best x, y tuple of a SubDataset."""
  if sub_dataset.y.shape[0] == 0:
    return None
  best_idx = jnp.argmax(sub_dataset.y)
  best_datapoint = (sub_dataset.x[best_idx],
                    sub_dataset.y[best_idx])
  return best_datapoint


def retrain_model(model: gp.GP,
                  sub_dataset_key: Union[int, str],
                  random_key: Optional[jax.Array] = None,
                  get_params_path: Optional[Callable[[Any], Any]] = None,
                  callback: Optional[Callable[[Any], Any]] = None):
  """Retrain the model with more observations on sub_dataset.

  Args:
    model: gp.GP.
    sub_dataset_key: key of the sub_dataset for testing in dataset.
    random_key: random state for jax.random, to be used for training.
    get_params_path: optional function handle that returns params path.
    callback: optional callback function for loggin of training steps.
  """
  retrain_condition = 'retrain' in model.params.config and model.params.config[
      'retrain'] > 0 and model.dataset[sub_dataset_key].x.shape[0] > 0
  if not retrain_condition:
    return
  if model.params.config['objective'] in [obj.regkl, obj.regeuc]:
    raise ValueError('Objective must include NLL to retrain.')
  max_training_step = model.params.config['retrain']
  logging.info(
      msg=('Retraining with max_training_step = '
           f'{max_training_step}.'))
  model.params.config['max_training_step'] = max_training_step
  model.train(
      random_key, get_params_path=get_params_path, callback=callback)


def bayesopt(
    key: Any,
    model: gp.GP,
    sub_dataset_key: Union[int, str],
    query_oracle: Callable[[Any], Any],
    ac_func: Callable[..., jnp.ndarray],
    iters: int,
    input_sampler: Callable[..., jnp.ndarray],
) -> SubDataset:
  """Running simulated bayesopt on a set of pre-evaluated inputs x_queries.

  Args:
    key: Jax random state.
    model: gp.GP.
    sub_dataset_key: key of the sub_dataset for testing in dataset.
    query_oracle: evaluation function.
    ac_func: acquisition function handle (see acfun.py).
    iters: number of iterations in BayesOpt sequential queries.
    input_sampler: function for sampling inputs among which the initial point is
      chosen for acquisition function optimization.

  Returns:
    All observations after bayesopt in the form of an (x_observed, y_observed)
    tuple. These observations include those made before bayesopt.
  """
  input_dim = model.input_dim
  for i in range(iters):
    start_time = time.time()
    retrain_model(model, sub_dataset_key=sub_dataset_key)
    key, subkey = jax.random.split(key)
    x_samples = input_sampler(subkey, input_dim)
    if ac_func.__name__ == 'rand' or ac_func.__name__ == 'random_search':
      logging.info('Using random search for bayesopt.')
      key, subkey = jax.random.split(key)
      select_idx = jax.random.choice(subkey, x_samples.shape[0])
    else:
      evals = ac_func(
          model=model, sub_dataset_key=sub_dataset_key, x_queries=x_samples)
      select_idx = evals.argmax()
    x_init = x_samples[select_idx]

    def f(x):
      return -ac_func(
          model=model,
          sub_dataset_key=sub_dataset_key,
          x_queries=jnp.array([x])).flatten()[0]

    opt = jaxopt.ScipyBoundedMinimize(method='L-BFGS-B', fun=f)
    opt_ret = opt.run(
        x_init, bounds=[jnp.zeros(input_dim),
                        jnp.ones(input_dim)])
    eval_datapoint = opt_ret.params, query_oracle(opt_ret.params[None, :])
    logging.info(msg=f'{i}-th iter, x_init={x_init}, '
                 f'eval_datapoint={eval_datapoint}, '
                 f'elpased_time={time.time() - start_time}')
    model.update_sub_dataset(
        eval_datapoint, sub_dataset_key=sub_dataset_key, is_append=True)

  return model.dataset.get(sub_dataset_key,
                           SubDataset(jnp.empty(0), jnp.empty(0)))


def simulated_bayesopt(
    model: gp.GP,
    sub_dataset_key: Union[int, str],
    queried_sub_dataset: SubDataset,
    ac_func: Callable[..., jnp.ndarray],
    iters: int,
    random_key: Optional[jax.Array] = None,
    get_params_path: Optional[Callable[[Any], Any]] = None,
    callback: Optional[Callable[[Any], Any]] = None,
) -> SubDataset:
  """Running simulated bayesopt on a set of pre-evaluated inputs x_queries.

  Args:
    model: gp.GP.
    sub_dataset_key: key of the sub_dataset for testing in dataset.
    queried_sub_dataset: sub_dataset that can be queried.
    ac_func: acquisition function handle (see acfun.py).
    iters: number of iterations in BayesOpt sequential queries.
    random_key: random state for jax.random, to be used for training or random
      search.
    get_params_path: optional function handle that returns params path.
    callback: optional callback function for loggin of training steps.

  Returns:
    All observations after bayesopt in the form of an (x_observed, y_observed)
    tuple. These observations include those made before bayesopt.
  """
  for _ in range(iters):
    if random_key is not None:
      random_key, subkey = jax.random.split(random_key)
    else:
      subkey = None
    retrain_model(
        model,
        sub_dataset_key=sub_dataset_key,
        random_key=subkey,
        get_params_path=get_params_path,
        callback=callback)
    if ac_func.__name__ == 'rand' or ac_func.__name__ == 'random_search':
      logging.info('Using random search for bayesopt.')
      if random_key is None:
        raise ValueError('Must specify a random key for random search.')
      random_key, subkey = jax.random.split(random_key)
      select_idx = jax.random.choice(subkey, queried_sub_dataset.x.shape[0])
    else:
      evals = ac_func(
          model=model,
          sub_dataset_key=sub_dataset_key,
          x_queries=queried_sub_dataset.x)
      select_idx = evals.argmax()
    eval_datapoint = queried_sub_dataset.x[select_idx], queried_sub_dataset.y[
        select_idx]
    model.update_sub_dataset(
        eval_datapoint, sub_dataset_key=sub_dataset_key, is_append=True)

  return model.dataset.get(sub_dataset_key,
                           SubDataset(jnp.empty(0), jnp.empty(0)))


def run_bayesopt(
    dataset: defs.AllowedDatasetTypes,
    sub_dataset_key: str,
    queried_sub_dataset: Union[SubDataset, Callable[[Any], Any]],
    mean_func: Callable[..., jnp.ndarray],
    cov_func: Callable[..., jnp.ndarray],
    init_params: defs.GPParams,
    ac_func: Callable[..., jnp.ndarray],
    iters: int,
    warp_func: defs.WarpFuncType = None,
    init_random_key: Optional[jax.Array] = None,
    method: str = 'hyperbo',
    init_model: bool = False,
    data_loader_name: str = '',
    get_params_path: Optional[Callable[[Any], Any]] = None,
    callback: Optional[Callable[[Any, Any, Any], Any]] = None,
    save_retrain_model: bool = False,
):
  """Running bayesopt experiment with synthetic data.

  Args:
    dataset: a list of vx, vy pairs, i.e. [(vx, vy)_i], where vx is
      m_points_historical x d and vy is m_points_historical x n; n > 1 iff this
      is an aligned subdataset.
    sub_dataset_key: key of the sub_dataset for testing in dataset.
    queried_sub_dataset: sub_dataset that can be queried.
    mean_func: mean function handle that maps from (params, n x d input,
      warp_func) to an n dimensional mean vector. (see vector_map in mean.py for
      more details).
    cov_func: covariance function handle that maps from (params, n1 x d input1,
      n2 x d input2, wrap_func) to a n1 x n2 covariance matrix (see matrix_map
      in kernel.py for more details).
    init_params: initial GP parameters for inference.
    ac_func: acquisition function handle (see acfun.py).
    iters: Number of iterations in sequential bayesopt queries.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.
    init_random_key: random state for jax.random, to be used to initialize
      required parts of GPParams or re-training or random search.
    method: BO method.
    init_model: to initialize model if True; otherwise False.
    data_loader_name: data loader name, e.g. pd1, hpob.
    get_params_path: optional function handle that returns params path.
    callback: optional callback function for loggin of training steps.
    save_retrain_model: save retrained model iif True.

  Returns:
    All observations in (x, y) pairs returned by the bayesopt strategy and all
    the best query point as in (x, y) pair. Model params.
  """
  logging.info(msg=f'run_synthetic is using method {method}.')
  if method in const.USE_HGP:
    model_class = gp.HGP
    init_params.config.update({
        'objective': 'nll',
        'method': 'slice_sample',
        'burnin': 50,
        'nsamples': 50,
        'priors': priors.DEFAULT_PRIORS,
    })
  else:
    model_class = gp.GP

  model = model_class(
      dataset=dataset,
      mean_func=mean_func,
      cov_func=cov_func,
      params=init_params,
      warp_func=warp_func)
  key = init_random_key
  if init_model:
    assert init_random_key is not None, ('Cannot initialize with '
                                         'init_random_key == None.')
    key, subkey = jax.random.split(key)
    model.initialize_params(subkey)
    # Infer GP parameters.
    key, subkey = jax.random.split(key)
    model.train(subkey, get_params_path, callback=callback)
  else:
    key, subkey = jax.random.split(key)
    model.rng = subkey
  if isinstance(queried_sub_dataset, SubDataset):
    best_query = get_best_datapoint(queried_sub_dataset)
    sub_dataset = simulated_bayesopt(
        model=model,
        sub_dataset_key=sub_dataset_key,
        queried_sub_dataset=queried_sub_dataset,
        ac_func=ac_func,
        iters=iters,
        random_key=key,
        get_params_path=get_params_path if save_retrain_model else None,
        callback=callback if save_retrain_model else None)
    return (sub_dataset.x,
            sub_dataset.y), best_query, model.params
  else:
    if data_loader_name not in INPUT_SAMPLERS:
      raise NotImplementedError(
          f'Input sampler for {data_loader_name} not found.')
    sub_dataset = bayesopt(
        key=key,
        model=model,
        sub_dataset_key=sub_dataset_key,
        query_oracle=queried_sub_dataset,
        ac_func=ac_func,
        iters=iters,
        input_sampler=INPUT_SAMPLERS[data_loader_name])
    return (sub_dataset.x, sub_dataset.y), None, model.params


def _onehot_matrix(shape, idx) -> np.ndarray:
  """Each row is a one-hot vector with idx-th element equal to 1."""
  zeros = np.zeros(shape)
  zeros[:, idx] = 1
  return zeros


def _subdataset_to_arrays(ds: SubDataset, dataset_id: int,
                          num_datasets: int) -> Tuple[np.ndarray, np.ndarray]:
  onehot = _onehot_matrix((ds.y.shape[0], num_datasets), dataset_id)
  return np.concatenate([ds.x, onehot], axis=1), ds.y  # pytype: disable=bad-return-type  # jax-ndarray


@dataclasses.dataclass
class _XYPair:
  """Helper class to keep x,y pair in sync."""
  x: np.ndarray
  y: np.ndarray

  def append_xy(self, other, idx: int) -> None:
    self.x = np.concatenate([self.x, other.x[idx:idx + 1, :]], axis=0)
    self.y = np.concatenate([self.y, other.y[idx:idx + 1, :]], axis=0)

  def delete(self, idx: int) -> None:
    self.x = np.delete(self.x, idx, 0)
    self.y = np.delete(self.y, idx, 0)

  def concat(self, other) -> '_XYPair':
    return _XYPair(
        x=np.concatenate([self.x, other.x]),
        y=np.concatenate([self.y, other.y]))

  def empty_like(self) -> '_XYPair':
    return _XYPair(
        x=np.zeros(0, self.x.shape[1]), y=np.zeros(0, self.y.shape[1]))

  @property
  def size(self):
    return self.x.shape[0]


