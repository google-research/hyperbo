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

"""Load or generate data for learning priors in BayesOpt."""

import functools
import itertools
import pickle

from absl import logging
from hyperbo.basics import data_utils
from hyperbo.basics import definitions as defs
from hyperbo.gp_utils import gp

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tensorflow.io import gfile

partial = functools.partial

SubDataset = defs.SubDataset

PD1 = {
    ('phase0', 'matched'): '../pd1/pd1_matched_phase0_results.jsonl',
    ('phase1', 'matched'): '../pd1/pd1_matched_phase1_results.jsonl',
    ('phase0', 'unmatched'): '../pd1/pd1_unmatched_phase0_results.jsonl',
    ('phase1', 'unmatched'): '../pd1/pd1_unmatched_phase1_results.jsonl',
}


def get_aligned_dataset(trials, study_identifier, labels, verbose=True):
  """Get aligned dataset from processed trials from get_dataset.

  Args:
    trials: pandas.DataFrame that stores all the trials.
    study_identifier: a label that uniquely identifies each study group.
    labels: labels of parameters and an eval metric (the last one).
    verbose: print info about data if True.

  Returns:
    aligned_dataset: Dict[str, SubDataset], mapping from aligned dataset names
    to an aligned SubDataset, with n x d input x, n x m evals y and non-empty
    aligned field with concatenated m study group names and aligned_suffix.
  """
  aligned_dataset = {}
  trials = trials[trials['aligned']]
  for aligned_suffix in trials['aligned_suffix'].unique():
    aligned_trials = trials[trials['aligned_suffix'] == aligned_suffix]
    aligned_groups = aligned_trials[study_identifier].unique()
    pivot_df = aligned_trials.pivot(
        index=labels[:-1], columns=study_identifier, values=labels[-1])
    nan_groups = [
        c for c in pivot_df.columns if pivot_df[c].isna().values.any()
    ]
    combnum = min(3, len(nan_groups) + 1, len(aligned_groups) - 1)
    for groups in itertools.chain(
        *[itertools.combinations(nan_groups, r) for r in range(combnum)]):
      remain_groups = [sg for sg in aligned_groups if sg not in groups]
      if groups:
        index = np.all([pivot_df[sg].isnull() for sg in groups], axis=0)
        sub_df = pivot_df.loc[index, remain_groups].dropna().reset_index()
      else:
        sub_df = pivot_df.dropna().reset_index()
      if sub_df.shape[0] > 0:
        if verbose:
          print('removed groups: ', groups)
          print('remaining groups: ', remain_groups)
          print('sub_df: ', sub_df.shape)
        aligned_key = ';'.join(list(groups) + [aligned_suffix])
        xx = jnp.array(sub_df[labels[:-1]])
        yy = jnp.array(sub_df[remain_groups])
        aligned_dataset[aligned_key] = SubDataset(
            x=xx, y=yy, aligned=';'.join(remain_groups + [aligned_suffix]))
  msg = f'aligned dataset: {jax.tree_map(jnp.shape, aligned_dataset)}'
  logging.info(msg=msg)
  if verbose:
    print(msg)
  return aligned_dataset


def get_dataset(trials, study_identifier, labels, verbose=True):
  """Get dataset from a dataframe.

  Args:
    trials: pandas.DataFrame that stores all the trials.
    study_identifier: a label that uniquely identifies each study group.
    labels: labels of parameters and an eval metric (the last one).
    verbose: print info about data if True.

  Returns:
    dataset: Dict[str, SubDataset], mapping from study group to a SubDataset.
  """
  study_groups = trials[study_identifier].unique()
  dataset = {}
  for sg in study_groups:
    study_trials = trials.loc[trials[study_identifier] == sg, labels]
    xx = jnp.array(study_trials[labels[:-1]])
    yy = jnp.array(study_trials[labels[-1:]])
    dataset[sg] = SubDataset(x=xx, y=yy)
  msg = f'dataset before align: {jax.tree_map(jnp.shape, dataset)}'
  logging.info(msg)
  if verbose:
    print(msg)
  return dataset


def sample_sub_dataset(key,
                       trials,
                       study_identifier,
                       labels,
                       p_observed=0.,
                       verbose=True,
                       sub_dataset_key=None):
  """Sample a sub-dataset from trials dataframe.

  Args:
    key: random state for jax.random.
    trials: pandas.DataFrame that stores all the trials.
    study_identifier: a label that uniquely identifies each study group.
    labels: labels of parameters and an eval metric (the last one).
    p_observed: percentage of data that is observed.
    verbose: print info about data if True.
    sub_dataset_key: sub_dataset name to be queried.

  Returns:
    trials: remaining trials after removing the sampled sub-dataset.
    sub_dataset_key: study group key for testing in dataset.
    queried_sub_dataset: SubDataset to be queried.
  """
  test_study_key, observed_key = jax.random.split(key, 2)
  if sub_dataset_key is None:
    study_groups = trials[study_identifier].unique()
    sub_dataset_id = jax.random.choice(test_study_key, len(study_groups))
    sub_dataset_key = study_groups[sub_dataset_id]
  else:
    study_groups = trials[study_identifier].unique()
    if sub_dataset_key not in study_groups:
      raise ValueError(f'{sub_dataset_key} must be in dataframe.')

  queried_trials = trials[trials[study_identifier] == sub_dataset_key].sample(
      frac=1. - p_observed, replace=False, random_state=observed_key[0])
  trials = trials.drop(queried_trials.index)

  xx = jnp.array(queried_trials[labels[:-1]])
  yy = jnp.array(queried_trials[labels[-1:]])
  queried_sub_dataset = SubDataset(x=xx, y=yy)

  msg = (f'removed study={sub_dataset_key}  '
         f'removed study shape: x-{queried_sub_dataset.x.shape}, '
         f'y-{queried_sub_dataset.y.shape}')
  logging.info(msg)
  if verbose:
    print(msg)

  return trials, sub_dataset_key, queried_sub_dataset


def process_dataframe(
    key,
    trials,
    study_identifier,
    labels,
    p_observed=0.,
    maximize_metric=True,
    warp_func=None,
    verbose=True,
    sub_dataset_key=None,
    num_remove=0,
    p_remove=0.,
):
  """Process a dataframe and return needed info for an experiment.

  Args:
    key: random state for jax.random or sub_dataset_key to be queried.
    trials: pandas.DataFrame that stores all the trials.
    study_identifier: a label that uniquely identifies each study group.
    labels: labels of parameters and an eval metric (the last one).
    p_observed: percentage of data that is observed.
    maximize_metric: a boolean indicating if higher values of the eval metric
      are better or not. If maximize_metric is False and there is no warping for
      the output label, we negate all outputs.
    warp_func: mapping from label names to warpping functions.
    verbose: print info about data if True.
    sub_dataset_key: sub_dataset name to be queried.
    num_remove: number of sub-datasets to remove.
    p_remove: proportion of data to be removed.

  Returns:
    dataset: Dict[str, SubDataset], mapping from study group to a SubDataset.
    sub_dataset_key: study group key for testing in dataset.
    queried_sub_dataset: SubDataset to be queried.
  """
  trials = trials[[study_identifier] + labels +
                  ['aligned', 'aligned_suffix']].copy(deep=True)
  trials = trials.dropna()
  if verbose:
    print('trials: ', trials.shape)
  if not warp_func:
    warp_func = {}
  logging.info(msg=f'warp_func = {warp_func}')
  if labels[-1] not in warp_func and not maximize_metric:
    warp_func[labels[-1]] = lambda x: -x
  for la, fun in warp_func.items():
    if la in labels:
      trials.loc[:, la] = fun(trials.loc[:, la])

  key, subkey = jax.random.split(key)
  trials, sub_dataset_key, queried_sub_dataset = sample_sub_dataset(
      key=subkey,
      trials=trials,
      study_identifier=study_identifier,
      labels=labels,
      p_observed=p_observed,
      verbose=verbose,
      sub_dataset_key=sub_dataset_key)

  for _ in range(num_remove):
    key, subkey = jax.random.split(key)
    removed_sub_dataset_key = None
    sub_dataset_key_split = sub_dataset_key.split(',')
    if len(sub_dataset_key_split) > 1:
      task_dataset_name = sub_dataset_key_split[1]
      study_groups = trials[study_identifier].unique()
      for s in study_groups:
        if task_dataset_name in s:
          removed_sub_dataset_key = s
    trials, _, _ = sample_sub_dataset(
        key=subkey,
        trials=trials,
        study_identifier=study_identifier,
        labels=labels,
        p_observed=p_observed,
        verbose=verbose,
        sub_dataset_key=removed_sub_dataset_key)
  if p_remove > 0:
    key, subkey = jax.random.split(key)
    removed_trials = trials.sample(
        frac=p_remove, replace=False, random_state=subkey[0])
    trials = trials.drop(removed_trials.index)

  dataset = get_dataset(
      trials=trials,
      study_identifier=study_identifier,
      labels=labels,
      verbose=verbose)

  aligned_dataset = get_aligned_dataset(
      trials=trials,
      study_identifier=study_identifier,
      labels=labels,
      verbose=verbose)
  dataset.update(aligned_dataset)
  return dataset, sub_dataset_key, queried_sub_dataset


def pd1(key,
        p_observed,
        verbose=True,
        sub_dataset_key=None,
        input_warp=True,
        output_log_warp=True,
        num_remove=0,
        metric_name='best_valid/error_rate',
        p_remove=0.):
  """Load PD1(Nesterov) from init2winit and pick a random study as test function.

  For matched dataframes, we set `aligned` to True in its trials and reflect it
  in its corresponding SubDataset.
  The `aligned` value and sub-dataset key has suffix aligned_suffix, which is
  its phase identifier.

  Args:
    key: random state for jax.random.
    p_observed: percentage of data that is observed.
    verbose: print info about data if True.
    sub_dataset_key: sub_dataset name to be queried.
    input_warp: apply warping to data if True.
    output_log_warp: use log warping for output.
    num_remove: number of sub-datasets to remove.
    metric_name: name of metric.
    p_remove: proportion of data to be removed.

  Returns:
    dataset: Dict[str, SubDataset], mapping from study group to a SubDataset.
    sub_dataset_key: study group key for testing in dataset.
    queried_sub_dataset: SubDataset to be queried.
  """
  all_trials = []
  for k, v in PD1.items():
    with gfile.GFile(v, 'r') as f:
      trials = pd.read_json(f, orient='records', lines=True)
      trials.loc[:, 'aligned'] = (k[1] == 'matched')
      trials.loc[:, 'aligned_suffix'] = k[0]
      all_trials.append(trials)
  trials = pd.concat(all_trials)
  labels = [
      'hps.lr_hparams.decay_steps_factor', 'hps.lr_hparams.initial_value',
      'hps.lr_hparams.power', 'hps.opt_hparams.momentum', metric_name
  ]
  warp_func = {}
  if input_warp:
    warp_func = {
        'hps.opt_hparams.momentum': lambda x: np.log(1 - x),
        'hps.lr_hparams.initial_value': np.log,
    }
  if output_log_warp:
    warp_func['best_valid/error_rate'] = lambda x: -np.log(x + 1e-10)

  return process_dataframe(
      key=key,
      trials=trials,
      study_identifier='study_group',
      labels=labels,
      p_observed=p_observed,
      maximize_metric=False,
      warp_func=warp_func if input_warp else None,
      verbose=verbose,
      sub_dataset_key=sub_dataset_key,
      num_remove=num_remove,
      p_remove=p_remove)


def _deduplicate(x, y, dataset_name, verbose=True):
  """Deduplicates x values, keeping the ones with highest y."""
  # Sort by decreasing Y values: deduplication keeps points with best rewards.
  sorted_xy = list(zip(*sorted(zip(x, y), key=lambda xy: xy[1], reverse=True)))
  x = np.array(sorted_xy[0])
  y = np.array(sorted_xy[1])
  _, idx = np.unique(x, axis=0, return_index=True)
  if verbose:
    print(
        f'Removed {x.shape[0] - len(idx)} duplicated points from {dataset_name}'
    )
  return x[idx, :], y[idx, :]


def _normalize_maf_dataset(maf_dataset, num_hparams, neg_error_to_accuracy):
  """Project the hparam values to [0, 1], and optionally convert y values.

  Args:
    maf_dataset: a dictionary of the format
      `{subdataset_name: {"X": ..., "Y":...}}`.
    num_hparams: the number of different hyperparameters being optimized.
    neg_error_to_accuracy: whether to transform y values that correspond to
      negative error rates to accuracy values.

  Returns:
    `maf_dataset` with normalized "X", and maybe "Y", values.
  """
  min_vals = np.ones(num_hparams) * np.inf
  max_vals = -np.ones(num_hparams) * np.inf

  for k, subdataset in maf_dataset.items():
    min_vals = np.minimum(min_vals, np.min(subdataset['X'], axis=0))
    max_vals = np.maximum(max_vals, np.max(subdataset['X'], axis=0))

  for k in maf_dataset:
    maf_dataset[k]['X'] = (maf_dataset[k]['X'] - min_vals) / (
        max_vals - min_vals)

    if neg_error_to_accuracy:
      maf_dataset[k]['Y'] = 1 + maf_dataset[k]['Y']
  return maf_dataset


def process_pd1_for_maf(outfile_path,
                        min_num_points,
                        input_warp,
                        output_log_warp,
                        neg_error_to_accuracy,
                        enforce_same_size_subdatasets,
                        verbose=True):
  """Store the pd1 dataset on cns in a MAF baseline-friendly format.

  Args:
    outfile_path: cns path where the pickled output is stored.
    min_num_points: Minimum number of points that must be in a subdataset to
      keep it.
    input_warp: apply warping to data if True.
    output_log_warp: use log warping for output.
    neg_error_to_accuracy: whether to transform y values that correspond to
      negative error rates to accuracy values. Cannot be True if output_log_warp
      is True.
    enforce_same_size_subdatasets: whether to to only keep `n` points of each
      subdataset, where `n` is the size of the smallest remaining dataset.
    verbose: print deduplication info about if True.
  """
  if output_log_warp and neg_error_to_accuracy:
    raise ValueError('Cannot transform y-values when the pd1 outputs are '
                     'log-warped!')

  key = jax.random.PRNGKey(0)
  dataset, _, _ = pd1(
      key, p_observed=1, input_warp=input_warp, output_log_warp=output_log_warp)

  num_hparams = dataset[list(dataset.keys())[0]].x.shape[1]
  excluded_subdatasets = ['imagenet_resnet50,imagenet,resnet,resnet50,1024']

  # Load and deduplicate data.
  maf_dataset = {}
  for k, subdataset in dataset.items():
    if subdataset.aligned is None and k not in excluded_subdatasets:
      x, y = _deduplicate(
          np.array(subdataset.x),
          np.array(subdataset.y),
          dataset_name=k,
          verbose=verbose)
      if x.shape[0] > min_num_points:
        maf_dataset[k] = dict(X=x, Y=y)

  if enforce_same_size_subdatasets:
    min_subdataset_size = min(
        [maf_dataset[k]['X'].shape[0] for k in maf_dataset])
    for k, subdataset in maf_dataset.items():
      x, y = maf_dataset[k]['X'], maf_dataset[k]['Y']
      maf_dataset[k] = dict(
          X=x[:min_subdataset_size, :], Y=y[:min_subdataset_size, :])
  maf_dataset = _normalize_maf_dataset(
      maf_dataset,
      num_hparams=num_hparams,
      neg_error_to_accuracy=neg_error_to_accuracy)

  data_utils.log_dataset(maf_dataset)
  with gfile.Open(outfile_path, 'wb') as f:
    pickle.dump(maf_dataset, f, pickle.HIGHEST_PROTOCOL)


  logging.info(
      msg=f'For HPOB dataset, finite = {finite}; surrogate = {surrogate}.')
  handler = hpob_handler.HPOBHandler(
      root_dir=hpob.ROOT_DIR, mode='v3', surrogates_dir=hpob.SURROGATES_DIR)
  if isinstance(search_space_index, str):
    search_space = search_space_index
  elif isinstance(search_space_index, int):
    spaces = list(handler.meta_train_data.keys())
    spaces.sort()
    search_space = spaces[search_space_index]
  else:
    raise ValueError('search_space_index must be str or int.')
  if isinstance(test_dataset_id_index, str):
    test_dataset_id = test_dataset_id_index
  elif isinstance(test_dataset_id_index, int):
    test_dataset_ids = list(handler.meta_test_data[search_space].keys())
    test_dataset_ids.sort()
    test_dataset_id = test_dataset_ids[test_dataset_id_index]
  else:
    raise ValueError('test_dataset_id_index must be str or int.')
  dataset = {}
  if aligned:
    dataset_id = list(handler.meta_train_data[search_space].keys())[0]
    train_x = np.array(handler.meta_train_data[search_space][dataset_id]['X'])
    input_dim = train_x.shape[1]
    key, subkey = jax.random.split(key, 2)
    train_x = sample_hpob_inputs(subkey, input_dim, wild_card_train)
    aligned_train_y = []
    for dataset_id in handler.meta_train_data[search_space]:
      hpob_oracle = hpob.HPOBContainer(handler).get_experimenter(
          search_space, dataset_id).EvaluateArray
      train_y = hpob_oracle(train_x)
      aligned_train_y.append(train_y)
    aligned_train_y = jnp.array(aligned_train_y).T
    dataset['aligned'] = SubDataset(x=train_x, y=aligned_train_y, aligned='all')
  else:
    for dataset_id in handler.meta_train_data[search_space]:
      train_x = np.array(handler.meta_train_data[search_space][dataset_id]['X'])
      if surrogate:
        if wild_card_train:
          input_dim = train_x.shape[1]
          key, subkey = jax.random.split(key, 2)
          train_x = sample_hpob_inputs(subkey, input_dim, wild_card_train)
        hpob_oracle = hpob.HPOBContainer(handler).get_experimenter(
            search_space, dataset_id).EvaluateArray
        train_y = hpob_oracle(train_x)[:, None]
      else:
        train_y = np.array(
            handler.meta_train_data[search_space][dataset_id]['y'])
      dataset[dataset_id] = SubDataset(x=train_x, y=train_y)
  if test_seed in ['test0', 'test1', 'test2', 'test3', 'test4']:
    init_index = handler.bo_initializations[search_space][test_dataset_id][
        test_seed]
    test_x = np.array(
        handler.meta_test_data[search_space][test_dataset_id]['X'])
    test_y = np.array(
        handler.meta_test_data[search_space][test_dataset_id]['y'])
    if finite:
      y_init = test_y[init_index]
    else:
      surrogate_name = 'surrogate-' + search_space + '-' + test_dataset_id
      y_min = handler.surrogates_stats[surrogate_name]['y_min']
      y_max = handler.surrogates_stats[surrogate_name]['y_max']
      y_init = np.clip(test_y[init_index], y_min, y_max)
    dataset[test_dataset_id] = SubDataset(x=test_x[init_index], y=y_init)
  else:
    test_x = np.empty((0, train_x.shape[1]))
    test_y = np.empty((0, 1))
  if finite:
    return dataset, test_dataset_id, SubDataset(x=test_x, y=test_y)
  else:
    hpob_oracle = hpob.HPOBContainer(handler).get_experimenter(
        search_space, test_dataset_id).EvaluateArray
    return dataset, test_dataset_id, hpob_oracle


def pd2(key,
        p_observed,
        verbose=True,
        sub_dataset_key=None,
        input_warp=True,
        output_log_warp=True,
        num_remove=0,
        metric_name='best_valid/error_rate',
        p_remove=0.):
  """Load PD2(Adam) data from init2winit and pick a random study as test function.

  For matched dataframes, we set `aligned` to True in its trials and reflect it
  in its corresponding SubDataset.
  The `aligned` value and sub-dataset key has suffix aligned_suffix, which is
  its phase identifier.

  Args:
    key: random state for jax.random.
    p_observed: percentage of data that is observed.
    verbose: print info about data if True.
    sub_dataset_key: sub_dataset name to be queried.
    input_warp: apply warping to data if True.
    output_log_warp: use log warping for output.
    num_remove: number of sub-datasets to remove.
    metric_name: name of metric.
    p_remove: proportion of data to be removed.

  Returns:
    dataset: Dict[str, SubDataset], mapping from study group to a SubDataset.
    sub_dataset_key: study group key for testing in dataset.
    queried_sub_dataset: SubDataset to be queried.
  """
  all_trials = []
  for k, v in PD2.items():
    with gfile.Open(v, 'rb') as f:
      trials = pickle.load(f)
      trials.loc[:, 'aligned'] = (k[1] == 'matched')
      trials.loc[:, 'aligned_suffix'] = k[0]
      all_trials.append(trials)
  trials = pd.concat(all_trials)
  labels = [
      'hps.lr_hparams.decay_steps_factor', 'hps.lr_hparams.initial_value',
      'hps.lr_hparams.power', 'hps.opt_hparams.beta1',
      'hps.opt_hparams.epsilon', metric_name
  ]
  warp_func = {}
  if input_warp:
    warp_func = {
        'hps.opt_hparams.beta1': lambda x: np.log(1 - x),
        'hps.lr_hparams.initial_value': np.log,
        'hps.opt_hparams.epsilon': np.log,
    }
  if output_log_warp:
    warp_func['best_valid/error_rate'] = lambda x: -np.log(x + 1e-10)

  return process_dataframe(
      key=key,
      trials=trials,
      study_identifier='study_group',
      labels=labels,
      p_observed=p_observed,
      maximize_metric=False,
      warp_func=warp_func if input_warp else None,
      verbose=verbose,
      sub_dataset_key=sub_dataset_key,
      num_remove=num_remove,
      p_remove=p_remove)


def grid2020(key,
             p_observed,
             verbose=True,
             sub_dataset_key=None,
             input_warp=True,
             output_log_warp=True,
             num_remove=0,
             p_remove=0.):
  """Load GRID_2020 data from init2winit and pick a random study as test function.

  Args:
    key: random state for jax.random.
    p_observed: percentage of data that is observed.
    verbose: print info about data if True.
    sub_dataset_key: sub_dataset name to be queried.
    input_warp: apply warping to data if True.
    output_log_warp: use log warping for output.
    num_remove: number of sub-datasets to remove.
    p_remove: proportion of data to be removed.

  Returns:
    dataset: Dict[str, SubDataset], mapping from study group to a SubDataset.
    sub_dataset_key: study group key for testing in dataset.
    queried_sub_dataset: SubDataset to be queried.
  """
  experiment_df, failed_trials = data_loader.parallel_load_trials_in_directories(
      GRID2020, 100)
  logging.info(msg=f'Loaded trials shape: {experiment_df.shape}'
               f'Failed trials len: {len(failed_trials)}')
  experiment_df = df_utils.add_best_eval_columns(
      experiment_df, ['valid/ce_loss', 'valid/error_rate'])
  experiment_df.loc[:, 'aligned'] = True
  experiment_df.loc[:, 'aligned_suffix'] = ''
  labels = [
      'hps.opt_hparams.momentum', 'hps.lr_hparams.initial_learning_rate',
      'hps.lr_hparams.power', 'hps.lr_hparams.decay_steps_factor',
      'best_valid/error_rate'
  ]
  warp_func = {}
  if input_warp:
    warp_func = {
        'hps.opt_hparams.momentum': lambda x: np.log(1 - x),
        'hps.lr_hparams.initial_learning_rate': np.log,
    }
  if output_log_warp:
    warp_func['best_valid/error_rate'] = lambda x: -np.log(x + 1e-10)
  return process_dataframe(
      key=key,
      trials=experiment_df,
      study_identifier='dataset',
      labels=labels,
      p_observed=p_observed,
      maximize_metric=False,
      warp_func=warp_func,
      verbose=verbose,
      sub_dataset_key=sub_dataset_key,
      num_remove=num_remove,
      p_remove=p_remove)




def random(key,
           mean_func,
           cov_func,
           params,
           dim,
           n_observed,
           n_queries,
           n_func_historical=0,
           m_points_historical=0,
           warp_func=None):
  """Generate random historical data and observed data for current function.

  Args:
    key: random state for jax.random.
    mean_func: mean function handle that maps from (params, n x d input,
      warp_func) to an n dimensional mean vector. (see vector_map in mean.py for
      more details).
    cov_func: covariance function handle that maps from (params, n1 x d input1,
      n2 x d input2, wrap_func) to a n1 x n2 covariance matrix (see matrix_map
      in kernel.py for more details).
    params: parameters for covariance, mean, and noise variance.
    dim: input dimension d.
    n_observed: number of observed data points for the queried function.
    n_queries: total number of data points that can be queried.
    n_func_historical: number of historical functions. These functions are
      observed before bayesopt on the new queried function.
    m_points_historical: number of points for each historical function.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.

  Returns:
    dataset: Dict[str, SubDataset], mapping from study group to a SubDataset.
    sub_dataset_key: study group key for testing in dataset.
    queried_sub_dataset: SubDataset to be queried.
  """
  x_key, y_key, historical_key = jax.random.split(key, 3)

  # Generate historical data.
  hist_keys = jax.random.split(historical_key, n_func_historical)
  dataset = {}
  for i in range(n_func_historical):
    x_hist_key, y_hist_key = jax.random.split(hist_keys[i], 2)
    vx = jax.random.uniform(x_hist_key, (m_points_historical, dim))
    vy = gp.sample_from_gp(
        y_hist_key, mean_func, cov_func, params, vx, warp_func=warp_func)
    dataset[i] = SubDataset(x=vx, y=vy)

  # Generate observed data and possible queries for queried function.
  vx = jax.random.uniform(x_key, (n_observed + n_queries, dim))
  vy = gp.sample_from_gp(
      y_key, mean_func, cov_func, params, vx, warp_func=warp_func)
  x_queries, x_observed = vx[:n_queries], vx[n_queries:]
  y_queries, y_observed = vy[:n_queries], vy[n_queries:]
  dataset[n_func_historical] = SubDataset(x=x_observed, y=y_observed)
  queried_sub_dataset = SubDataset(x=x_queries, y=y_queries)
  return dataset, n_func_historical, queried_sub_dataset

