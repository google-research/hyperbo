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

"""Plot utils for Bayesopt results."""
import collections
import concurrent.futures
import logging
import os

from hyperbo.basics import params_utils
import utils  # local file import
import numpy as np
# For backward compatibility
plot_all = utils.plot_all


def decode_exp_key(exp_key, data_loader_name):
  """Decode exp_key from synthetic.py to method - list of result dict."""
  elements = exp_key.split('-')
  if data_loader_name == 'pd1':
    test_dataset_index, seed, mean_func_name, cov_func_name, mlp_features, objective, opt_method, max_training_step, batch_size, num_remove, p_observed, p_remove, _, _, acfun, method = elements
    return acfun, int(num_remove), test_dataset_index, '-'.join(
        (seed, mean_func_name, cov_func_name, mlp_features, objective,
         opt_method, max_training_step, batch_size, p_observed, p_remove,
         method))
  elif 'hpob' in data_loader_name:
    return
  else:
    raise NotImplementedError(f'{data_loader_name} Not Implemented.')


def run_in_parallel(function, list_of_kwargs_to_function, num_workers):
  """Run a function on a list of kwargs in parallel with ThreadPoolExecutor.

  Adapted from code by mlbileschi.
  Args:
    function: a function.
    list_of_kwargs_to_function: list of dictionary from string to argument
      value. These will be passed into `function` as kwargs.
    num_workers: int.

  Returns:
    list of return values from function.
  """
  if num_workers < 1:
    raise ValueError(
        'Number of workers must be greater than 0. Was {}'.format(num_workers))

  with concurrent.futures.ThreadPoolExecutor(num_workers) as executor:
    futures = []
    logging.info(
        'Adding %d jobs to process pool to run in %d parallel '
        'threads.', len(list_of_kwargs_to_function), num_workers)

    for kwargs in list_of_kwargs_to_function:
      f = executor.submit(function, **kwargs)
      futures.append(f)

    for f in concurrent.futures.as_completed(futures):
      if f.exception():
        # Propagate exception to main thread.
        raise f.exception()

  return [f.result() for f in futures]


def get_model(dirnm, unique_id, verbose, filenm='result.pkl', retry=True):
  """Get result from one bo run."""
  file = os.path.join(dirnm, filenm)
  res = params_utils.load_params(file, use_gpparams=False)
  if not res and not retry:
    if verbose:
      print(f'{filenm} ')
    return None

  workload_key = 'sub_dataset_key'
  if 'sub_dataset_key' not in res:
    if 'subdataset_key' in res:
      workload_key = 'subdataset_key'
    else:
      print(f'{filenm} wrong format.')
      print(f'res keys={res.keys()}')
      return None

  workload = str(res[workload_key])

  return (workload, unique_id), res['params_dict']


# TODO(wangzi): refactor functions that process the results.
def get_exp_result(dirnm,
                   unique_id,
                   verbose,
                   filenm='result.pkl',
                   retry=True,
                   maf=False):
  """Get result from one bo run."""
  file = os.path.join(dirnm, filenm)
  res = params_utils.load_from_file(file)
  # res = res[1]
  if not res and not retry:
    return None
  yy = res['observations'][1].flatten()
  yq = res['queries'][1].flatten()

  workload_key = 'sub_dataset_key'
  if 'sub_dataset_key' not in res:
    if 'subdataset_key' in res:
      workload_key = 'subdataset_key'
    else:
      print(f'{filenm} wrong format.')
      print(f'res keys={res.keys()}')
      return None

  workload = str(res[workload_key])
  if workload == 'imagenet_resnet50,imagenet,resnet,resnet50,1024':
    return None
  if maf:
    yy = 1 - yy  # error_rate
    yy = -np.log(yy + 1e-10)  # warped error_rate
    yq = 1 - yq  # error_rate
    yq = -np.log(yq + 1e-10)  # warped error_rate

  maxy = max(max(yy), max(yq))
  regret_array = [maxy - max(yy[:j + 1]) for j in range(len(yy))]
  if verbose:
    print(f'filenm={filenm}, \n'
          f'dirnm={dirnm}, \n'
          f'len(regret)={len(regret_array)}, \n'
          f'final regret={regret_array[-1]} \n')
  if maf and len(regret_array) < 100:
    print('')
    return None

  return (workload, unique_id), (regret_array, yy, maxy)


def add_regret_array(res):
  """Add regret array to result dict with observations."""
  yy = res['observations'][1].flatten()
  best_query_y = res['best_query'][1]
  # if output_log_warp:
  #   yy = output_warper_inverse(yy)
  #   best_query_y = output_warper_inverse(best_query_y)
  maxy = max(max(yy), best_query_y)
  regret_array = []
  maxy_tmp = -np.inf
  for j in range(len(yy)):
    maxy_tmp = max(maxy_tmp, yy[j])
    regret_array.append(maxy - maxy_tmp)
  res['regret_array'] = regret_array
  res['maxy'] = maxy
  return res


def process_results(results, verbose=True):
  """Get result from one bo run."""
  if not results:
    return None

  # def output_warper_inverse(y):
  #   return -np.exp(-y) + 1e-6 + 1.

  for exp_key, res in results.items():
    res = add_regret_array(res)
    if verbose:
      regret_array = res['regret_array']
      print(f'exp_key={exp_key}, \n'
            f'len(regret)={len(regret_array)}, \n'
            f'final regret={regret_array[-1]} \n')
  return results


def get_hpob_exp(kwarg, verbose=True):
  """Get result from one bo run."""
  filenm, unique_id = kwarg['filenm'], kwarg['unique_id']
  results = params_utils.load_params(
      filenm, use_gpparams=False, include_state=True)
  if not results:
    return None
  if not isinstance(results, tuple):
    print(results)
  results = results[1]

  def output_warper_inverse(y):
    return -np.exp(-y) + 1e-6 + 1.

  for exp, res in results.items():
    exp_key = exp[0]
    yy = res['observations'][1].flatten()
    best_query_y = res['best_query'][1]

    if 'output_log_warp' in exp_key:
      yy = output_warper_inverse(yy)
      best_query_y = output_warper_inverse(best_query_y)

    exp_key = '-'.join((res['search_space'], res['sub_dataset_key']))
    maxy = max(max(yy), best_query_y)
    regret_array = [maxy - max(yy[:j + 1]) for j in range(len(yy))]
    res['regret_array'] = regret_array
    res['yy'] = yy
    res['maxy'] = maxy
  if verbose:
    print(f'filenm={filenm}, \n'
          f'len(regret)={len(regret_array)}, \n'
          f'final regret={regret_array[-1]} \n')
    print((exp_key, unique_id), flush=True)
  return (exp_key, unique_id), results


def get_multi_hpob_exp(kwargs):
  res = []
  print(f'in multi exp - kwargs: {len(kwargs)}', flush=True)
  for kwarg in kwargs:
    res.append(get_hpob_exp(kwarg))
  print(f'in multi exp: {len(res)}', flush=True)
  return res


def hpob_results(kwargs,
                 verbose=False,
                 process_func=get_multi_hpob_exp,
                 n=100,
                 parallel=True):
  """Get a dict of results aggregated over n files in a directory.

  Args:
    kwargs: dict with field filenm and unique_id.
    verbose: print logging messages if True.
    process_func: function to process each result file.
    n: maximum number of sequential files to read for each parallel worker.
    parallel: use run_in_parallel if True; otherwise sequential.

  Returns:
    a dictionary mapping from the first returned item of process_func to the
    second returned item of process_func.
  """
  kwarg_list = []
  sub_list = []
  cnt = 0
  for kwarg in kwargs:
    kwarg['verbose'] = verbose
    sub_list.append(kwarg)
    cnt += 1
    if cnt % n == 0:
      kwarg_list.append({'kwargs': sub_list})
      sub_list = []
      cnt = 0
  if sub_list:
    kwarg_list.append({'kwargs': sub_list})
  results = []
  if parallel:
    results = run_in_parallel(process_func, kwarg_list,
                              min(len(kwargs) // n, 100))
  else:
    for kwarg in kwarg_list:
      r = process_func(**kwarg)
      results.append(r)

  result_list = []
  for sub_res in results:
    for r in sub_res:
      if r is not None:
        result_list.append(r)
  return dict(result_list)


def get_results(directory, n, verbose=False, process_func=get_exp_result):
  """Get a dict of results aggregated over n files in a directory.

  Args:
    directory: file directory of results.
    n: number of result files.
    verbose: print logging messages if True.
    process_func: function to process each result file.

  Returns:
    a dictionary mapping from the first returned item of process_func to the
    second returned item of process_func.
  """
  kwarg_list = []
  for i in range(n):
    kwarg = {
        'dirnm': os.path.join(directory, str(i + 1)),
        'unique_id': i,
        'verbose': verbose
    }
    kwarg_list.append(kwarg)
  results = run_in_parallel(process_func, kwarg_list, 100)
  results = [r for r in results if r is not None]
  return dict(results)


WORKLOAD2NAME = {
    'cifar10_wrn,cifar10,wide_resnet,wrn,2048':
        'CIFAR10 WRN 2048',
    'cifar10_wrn,cifar10,wide_resnet,wrn,256':
        'CIFAR10 WRN 256',
    'cifar100_wrn,cifar100,wide_resnet,wrn,2048':
        'CIFAR100 WRN 2048',
    'cifar100_wrn,cifar100,wide_resnet,wrn,256':
        'CIFAR100 WRN 256',
    'fashion_maxp_cnn,fashion_mnist,max_pooling_cnn,max_pool_relu,2048':
        'Fashion CNNPoolReLU 2048',
    'fashion_maxp_cnn,fashion_mnist,max_pooling_cnn,max_pool_relu,256':
        'Fashion CNNPoolReLU 256',
    'fashion_maxp_cnn,fashion_mnist,max_pooling_cnn,max_pool_tanh,2048':
        'Fashion CNNPoolTanh 2048',
    'fashion_maxp_cnn,fashion_mnist,max_pooling_cnn,max_pool_tanh,256':
        'Fashion CNNPoolTanh 256',
    'fashion_smpl_cnn,fashion_mnist,simple_cnn,simple_cnn,2048':
        'Fashion CNNReLU 2048',
    'fashion_smpl_cnn,fashion_mnist,simple_cnn,simple_cnn,256':
        'Fashion CNNReLU 256',
    # 'imagenet_resnet50,imagenet,resnet,resnet50,1024':
    #     'ImageNet ResNet50 1024',
    'imagenet_resnet50,imagenet,resnet,resnet50,256':
        'ImageNet ResNet50 256',
    'imagenet_resnet50,imagenet,resnet,resnet50,512':
        'ImageNet ResNet50 512',
    'lm1b_trfmr,lm1b,transformer,transformer,2048':
        'LM1B Transformer 2048',
    'mnist_maxp_cnn,mnist,max_pooling_cnn,max_pool_relu,2048':
        'MNIST CNNPoolReLU 2048',
    'mnist_maxp_cnn,mnist,max_pooling_cnn,max_pool_relu,256':
        'MNIST CNNPoolReLU 256',
    'mnist_maxp_cnn,mnist,max_pooling_cnn,max_pool_tanh,2048':
        'MNIST CNNPoolTanh 2048',
    'mnist_maxp_cnn,mnist,max_pooling_cnn,max_pool_tanh,256':
        'MNIST CNNPoolTanh 256',
    'mnist_simple_cnn,mnist,simple_cnn,simple_cnn,2048':
        'MNIST CNNReLU 2048',
    'mnist_simple_cnn,mnist,simple_cnn,simple_cnn,256':
        'MNIST CNNReLU 256',
    'svhn_noextra_wrn,svhn_no_extra,wide_resnet,wrn,1024':
        'SVHN WRN 1024',
    'svhn_noextra_wrn,svhn_no_extra,wide_resnet,wrn,256':
        'SVHN WRN 256',
    'uniref50_trfmr,uniref50,transformer,transformer,128':
        'Uniref50 Transformer 128',
    'wmt15_de_en_xfmr,translate_wmt,xformer_translate,xformer,64':
        'WMT XFormer 64',
}


def get_workload2result(res,
                        error_rate,
                        best_only=True,
                        use_name=True,
                        max_training_step=100):
  """Returns workload2result dict.

  Args:
    res: variable returned by get_results, dict mapping from method to
      teststudy2y_array dict.
    error_rate: use error rate instead of regret on warped output.
    best_only: only include the best regret or error rate instead of the entire
      sequence computed from BayesOpt.
    use_name: use workload names specified in WORKLOAD2NAME.
    max_training_step: maximum iteration to compute best error rate or regret.

  Returns:
    Dictionary mapping from workload to a result dictionary. The result
    dictionary maps from method name to a list of relevant experiment results:
    sequences or best of error rates or regrets.
  """
  workload2result = collections.defaultdict(dict)
  for method in res:
    teststudy2y_array = res[method]
    for wl, i in teststudy2y_array:
      wl = str(wl)
      if method not in workload2result[wl]:
        workload2result[wl][method] = []
      if error_rate:
        yy = teststudy2y_array[(wl, i)][1]
        yy = np.exp(-yy) - 1e-10
        if best_only:
          workload2result[wl][method].append(min(yy[:max_training_step]))
        else:
          workload2result[wl][method].append(yy)
      else:
        regret = teststudy2y_array[(wl, i)][0]
        if best_only:  # get the regret in the last round of BO
          workload2result[wl][method].append(regret[max_training_step - 1])
        else:
          workload2result[wl][method].append(regret)
  if use_name:
    workload2result = {
        WORKLOAD2NAME[wl]: workload2result[wl] for wl in workload2result
    }
  return workload2result


def analyze_results(res, percentile=20, error_rate=True, max_training_step=100):
  """Analyze results for each workload and method.

  Args:
    res: variable returned by get_results.
    percentile: desired percentile to return metrics per workload and method.
    error_rate: use error rate instead of regret on warped output.
    max_training_step: maximum iteration to compute best error rate or regret.

  Returns:
    Dictionary mapping from workload to a result dictionary. The result
    dictionary maps from method name to 5 metrics (mean, std, lower percentile,
    median, upper percentile).
  """
  workload2result = get_workload2result(
      res, error_rate, best_only=True, max_training_step=max_training_step)
  for method in res:
    for wl in workload2result:
      if method in workload2result[wl]:
        final_result = np.array(workload2result[wl][method])
        lower, median, upper = np.percentile(
            final_result, [percentile, 50, 100 - percentile], axis=0)
        mean = np.mean(final_result, axis=0)
        std = np.std(final_result, axis=0)
        val = mean, std, lower, median, upper
      else:
        val = []
      workload2result[wl][method] = val
  return workload2result


def compute_workload2ref(workload2result,
                         methods,
                         trial=100,
                         ref_metric='median'):
  """Compute reference metric values for each workload.

  We can use the following two options for performance profile figures:
  (1) Using the median value of Random Search at 100 trials OR
  (2) Using the best median value across all methods at {25, 50} trials (or some
  other number that is small compared to the maximum number of trials we ran)

  Args:
    workload2result: dictionary mapping from workload to a result dictionary.
      The result dictionary maps from method name to a list of minimizing
      metrics.
    methods: list of method strings to compute the reference.
    trial: trial at which we compute the reference.
    ref_metric: the metric to set the reference value for each task. If
      ref_metric is 'median', we set it to be the best median value across all
      methods.

  Returns:
    dict mapping from workoad to reference metrics.
  """
  workload2ref = {}
  for wl in workload2result:
    y_arrays = []
    for method in methods:
      y_arrays += workload2result[wl][method]
    y_arrays = np.array(y_arrays)[:, :trial]
    if ref_metric == 'median':
      workload2ref[wl] = np.median(np.amin(y_arrays, 1))
    elif ref_metric == 'mean':
      workload2ref[wl] = np.mean(np.amin(y_arrays, 1))
    elif isinstance(ref_metric, float):
      workload2ref[wl] = ref_metric * min(y_arrays.flatten())
  return workload2ref


def get_method2fraction(workload2result, workload2ref, bo_iters=100):
  """Compute dict mapping from method to fraction."""
  method2fraction = collections.defaultdict(lambda: np.zeros(bo_iters))
  total = collections.defaultdict(lambda: 0)
  for wl in workload2result:
    for method, result in workload2result[wl].items():
      for yy in result:
        total[method] += 1
        for i in range(len(yy)):
          method2fraction[method][i] += 1 if min(
              yy[:i + 1]) <= workload2ref[wl] + 1e-6 else 0
  for method in method2fraction:
    method2fraction[method] = method2fraction[method] / total[method]
  return method2fraction
