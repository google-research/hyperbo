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

"""Objective functions for training a GP."""
import functools
import logging

from hyperbo.basics import linalg
from hyperbo.basics import params_utils
from hyperbo.gp_utils import utils
import jax.numpy as jnp
import jax.scipy.linalg as jspla

retrieve_params = params_utils.retrieve_params


def multivariate_normal_divergence(
    mean_func,
    cov_func,
    params,
    dataset,
    warp_func=None,
    distance=utils.kl_multivariate_normal,
):
  """Compute the multivariate normal divergence between the data and the model.

  The returned objective describes the distance between the multivariate normal
  specified by sample mean/covariance and the multivariate normal specified by
  the parameterized GP model.
  We support KL divergence as distance or squared Euclidean distance.

  Args:
    mean_func: mean function handle that maps from (params, n x d input,
      warp_func) to an n dimensional mean vector. (see vector_map in mean.py for
      more details).
    cov_func: covariance function handle that maps from (params, n1 x d input1,
      n2 x d input2, wrap_func) to a n1 x n2  covariance matrix (see matrix_map
      in kernel.py for more details).
    params: parameters for covariance, mean, and noise variance.
    dataset: Dict[Union[int, str], SubDataset], a dictionary mapping from key to
      SubDataset. For aligned sub-dataset, this function should only be used if
      each aligned sub-dataset only has (?, m) for y shape, where m > 1.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.
    distance: distance function; currently support utils.kl_multivariate_normal
      or utils.euclidean_multivariate_normal.

  Returns:
    The divergence value between two multivariate normal distributions, one is
    defined by the data and the other is defined by the GP model.
  """

  def compute_metric_per_sub_dataset(sub_dataset):
    """Compute the regularizer on a subset of dataset keys."""
    if sub_dataset.y.shape[0] == 0:
      return 0.
    mu_data = jnp.mean(sub_dataset.y, axis=1)
    cov_data = jnp.cov(sub_dataset.y, bias=True)
    mu_model = mean_func(params, sub_dataset.x, warp_func=warp_func).flatten()
    noise_variance, = retrieve_params(
        params, ['noise_variance'], warp_func=warp_func)
    cov_model = cov_func(
        params, sub_dataset.x,
        warp_func=warp_func) + jnp.eye(sub_dataset.x.shape[0]) * noise_variance

    return distance(
        mu0=mu_data,
        cov0=cov_data,
        mu1=mu_model,
        cov1=cov_model)

  total_val = 0.
  num_sub_datasets = 0
  for sub_dataset_key, sub_dataset in dataset.items():
    if sub_dataset.aligned is None:
      continue
    if sub_dataset.x.shape[0] == 0:
      continue
    if sub_dataset.y.shape[
        1] == 0 or sub_dataset.y.shape[0] != sub_dataset.x.shape[0]:
      raise ValueError(
          (f'dataset[{sub_dataset_key}].x has shape {sub_dataset.x.shape} '
           f'but dataset[{sub_dataset_key}].y has shape {sub_dataset.y.shape}'))
    total_val += compute_metric_per_sub_dataset(sub_dataset)
    num_sub_datasets += 1

  if num_sub_datasets == 0:
    return 0.
  return total_val / num_sub_datasets


multivariate_normal_euc_distance = functools.partial(
    multivariate_normal_divergence,
    distance=utils.euclidean_multivariate_normal)


def neg_log_marginal_likelihood(mean_func,
                                cov_func,
                                params,
                                dataset,
                                warp_func=None,
                                exclude_aligned=True,
                                return_key2nll=False,
                                use_cholesky=True):
  """Compute the negative of marginal likelihood of a (multi-task) GP.

  Args:
    mean_func: mean function handle that maps from (params, n x d input,
      warp_func) to an n dimensional mean vector. (see vector_map in mean.py for
      more details).
    cov_func: covariance function handle that maps from (params, n1 x d input1,
      n2 x d input2, wrap_func) to a n1 x n2  covariance matrix (see matrix_map
      in kernel.py for more details).
    params: parameters for covariance, mean, and noise variance.
    dataset: Dict[Union[int, str], SubDataset], a dictionary mapping from key to
      SubDataset.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.
    exclude_aligned: exclude sub-datasets that are aligned.
    return_key2nll: return total_nll together with the dictionary mapping from
      sub-dataset key to its corresponding nll value.
    use_cholesky: use cholesky to compute NLL if True; otherwise, use SVD. We
      need the SVD option due to numerical instability of cholesky when the
      kernel is positive definite but the covariance matrix is numerically low
      rank.

  Returns:
    Negative log marginal likelihood if return_key2nll is False; otherwise a
    tuple consisting of total nll and the sub-dataset key to nll dictionary.
  """

  def compute_nll_sub_dataset_cholesky(vx, vy):
    """Compute negative log likelihood for one sub dataset."""
    chol, kinvy, vy = linalg.solve_gp_linear_system(
        mean_func=mean_func,
        cov_func=cov_func,
        params=params,
        x=vx,
        y=vy,
        warp_func=warp_func)
    nll_val = jnp.sum(0.5 * jnp.dot(vy.T, kinvy) +
                      jnp.sum(jnp.log(jnp.diag(chol))) +
                      0.5 * len(vx) * jnp.log(2 * jnp.pi))
    return nll_val
  def compute_nll_sub_dataset_svd(vx, vy):
    """Compute negative log likelihood for one sub dataset."""
    vy, cov = linalg.compute_delta_y_and_cov(
        mean_func=mean_func,
        cov_func=cov_func,
        params=params,
        x=vx,
        y=vy,
        warp_func=warp_func)
    (u, s, v) = jspla.svd(cov)
    if s[-1] <= 0:
      logging.warning(msg=f'Covariance matrix is low rank. s = {s}')
    kinv = jnp.dot(v.T, jnp.dot(jnp.diag(s**-1), u.T))
    kinvy = jnp.dot(kinv, vy)
    nll_val = 0.5 * jnp.sum(
        jnp.dot(vy.T, kinvy)
        + jnp.sum(jnp.log(s))
        + len(vx) * jnp.log(2 * jnp.pi)
    )
    return nll_val

  total_nll = 0.
  key2nll = {}
  num_sub_datasets = 0
  for k, s in dataset.items():
    if exclude_aligned and s.aligned is not None:
      continue
    if s.x.shape[0] == 0:
      continue
    if use_cholesky:
      key2nll[k] = compute_nll_sub_dataset_cholesky(s.x, s.y)
    else:
      key2nll[k] = compute_nll_sub_dataset_svd(s.x, s.y)
    total_nll += key2nll[k]
    num_sub_datasets += 1
  if num_sub_datasets == 0:
    total_nll = 0.
  else:
    total_nll /= num_sub_datasets

  # We should really be including priors on the hyperparameters here.
  if 'priors' in params.config:
    for k in params.model:
      if k in params.config['priors']:
        log_prior_fn = params.config['priors'][k]
        val, = retrieve_params(params, [k], warp_func)
        log_prior_prob = log_prior_fn(val)
        logging.info(msg=f'log_prior_prob({k}={val}) = {log_prior_prob}')
        total_nll -= log_prior_prob
      else:
        logging.warning('No prior provided for param %s', k)
  if return_key2nll:
    return total_nll, key2nll
  return total_nll


nll = neg_log_marginal_likelihood
kl = multivariate_normal_divergence
ekl = kl
euc = multivariate_normal_euc_distance
regkl = kl
regeuc = euc


def add(*objectives):

  def added_objective(*args, **kwargs):
    return sum([obj(*args, **kwargs) for obj in objectives])

  return added_objective


def mul(c, obj):

  def multiplied_objective(*args, **kwargs):
    return c * obj(*args, **kwargs)

  return multiplied_objective


nll_regkl = lambda c: add(nll, mul(c, regkl))
nll_regeuc = lambda c: add(nll, mul(c, regeuc))

nll_regkl1 = nll_regkl(1.)
nll_regeuc1 = nll_regeuc(1.)
nll_regkl01 = nll_regkl(.1)
nll_regeuc01 = nll_regkl(.1)

nll_regkl10 = nll_regkl(10.)
nll_regeuc10 = nll_regkl(10.)
