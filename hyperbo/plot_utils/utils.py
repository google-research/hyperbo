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

"""Utils for plots related to hyperbo."""

import matplotlib.pyplot as plt
import numpy as np

COLORS = {
    'Rand': '#1f77b4',
    'STBO': '#7f7f7f',
    'STBOH': '#2ca02c',
    'MAF': '#ff7f0e',
    'MIMO': '#8c564b',
    'RFGP': '#e377c2',
    'H* NLL': '#d62728',
    'H* KL': '#6a3d9a',
    'H* NLLEUC': '#bcbd22',
    'H* NLLKL': '#17becf'
}


def plot_with_upper_lower(x,
                          line,
                          lower,
                          upper,
                          color='r',
                          ax=None,
                          set_xticks=False,
                          **plot_kwargs):
  """Plot mean and standard deviation with inputs x."""
  if ax is None:
    plt.figure()
    ax = plt.gca()
  ax.fill_between(x, lower, upper, alpha=.1, color=color)
  ax.plot(x, line, color=color, **plot_kwargs)
  if x is not None and set_xticks:
    ax.set_xticks(x)


def plot_array_mean_std(array, color, x=None, ax=None, axis=0, **plot_kwargs):
  """Plot experiment results stored in array."""
  mean, std = np.mean(array, axis=axis), np.std(array, axis=axis)
  if x is None:
    x = range(1, len(mean) + 1)
  plot_with_upper_lower(x, mean, mean - std, mean + std, color, ax,
                        **plot_kwargs)


def plot_array_median_percentile(array,
                                 color,
                                 x=None,
                                 ax=None,
                                 percentile=20,
                                 **plot_kwargs):
  """Plot experiment results stored in array."""
  lower, median, upper = np.percentile(
      array, [percentile, 50, 100 - percentile], axis=0)
  if x is None:
    x = range(1, len(median) + 1)
  plot_with_upper_lower(x, median, lower, upper, color, ax, **plot_kwargs)


def plot_all(label2array,
             ax,
             logscale=True,
             ylabel='Regret',
             xlabel='BO Iters',
             method='mean',
             colors=COLORS.copy(),
             **kwargs):
  """Plot all experiment results.

  Args:
    label2array: a dictionary with labels as keys and an array of results as
      values.
    ax: matplotlib.pyplot.axis.
    logscale: use log scale for y axis if True.
    ylabel: label for y axis.
    xlabel: label for x axis.
    method: plot mean and std, or median and percentile.
    colors: dictionary mapping from label to color.
    **kwargs: other plot arguments.
  """
  assert len(label2array) <= len(
      colors
  ), f'max number of lines to plot is {len(colors)} got {len(label2array)}'
  exp_types = label2array.keys()
  iteritems = []
  for label in exp_types:
    if label not in colors:
      iteritems = zip(list(colors.values())[:len(exp_types)], exp_types)
      print(f'Colors not assigned to {label}.')
      break
    else:
      iteritems += [(colors[label], label)]

  for color, label in iteritems:
    if label not in label2array or label2array[label] is None:
      continue
    y_array = np.array(label2array[label])
    if method == 'mean':
      plot_array_mean_std(y_array, ax=ax, label=label, color=color, **kwargs)
    elif method == 'median':
      plot_array_median_percentile(
          y_array, ax=ax, label=label, color=color, **kwargs)
    if logscale:
      ax.set_yscale('log')
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def set_violin_axis_style(ax, labels):
  """Set the style of an axis on a violin plot."""
  ax.xaxis.set_tick_params(direction='out')
  ax.xaxis.set_ticks_position('bottom')
  ax.set_xticks(np.arange(1, len(labels) + 1))
  ax.set_xticklabels(labels, rotation=45)
  ax.set_xlim(0.5, len(labels) + 0.5)


def plot_summary(labels,
                 label2array,
                 xlim=(1, 100),
                 ylim=None,
                 logscale=True,
                 ylabel='Regret',
                 xlabel='BO Iters',
                 method='mean',
                 title=None,
                 violin_trials=None,
                 violin_labels=None,
                 figsize=(24, 6),
                 colors=COLORS.copy(),
                 axes=None,
                 uppercenter_legend=True,
                 uppercenter_legend_ncol=3,
                 bbox_to_anchor=(0.5, 1.1),
                 **kwargs):
  """Plot a summary of results with options to add violin plots on slices.

  Args:
    labels: list of labels to be included in the plot.
    label2array: a dictionary with labels as keys and an array of results as
      values.
    xlim: a tuple of the new x-axis limits.
    ylim: a tuple of the new y-axis limits.
    logscale: use log scale for y axis if True.
    ylabel: label for y axis.
    xlabel: label for x axis.
    method: plot mean and std, or median and percentile.
    title: title of the plot.
    violin_trials: list of trials to plot violin plots on slices of the figure.
    violin_labels: list of lables to be included in violin plots.
    figsize: a tuple describing the size of the figure.
    colors: dictionary mapping from label to color.
    axes: list of matplotlib.pyplot.axis objects.
    uppercenter_legend: use an upper center legend if True.
    uppercenter_legend_ncol: number of columns for the upper center legend.
    bbox_to_anchor: bbox_to_anchor of the upper center legend.
    **kwargs: other plot arguments.
  """
  plt.figure(dpi=1500)
  if axes is None or len(axes) < len(violin_trials) + 1:
    _, axes = plt.subplots(
        nrows=1, ncols=len(violin_trials) + 1, figsize=figsize)
  plot_all({la: label2array.get(la, None) for la in labels},
           axes[0],
           logscale=logscale,
           ylabel=ylabel,
           xlabel=xlabel,
           method=method,
           colors=colors,
           **kwargs)
  axes[0].set_xlim(xlim)
  if uppercenter_legend:
    axes[0].legend(
        loc='upper center',
        bbox_to_anchor=bbox_to_anchor,
        ncol=uppercenter_legend_ncol,
        fancybox=True,
        shadow=True)
  else:
    axes[0].legend()
  if ylim:
    axes[0].set_ylim(ylim[0], ylim[1])
  if title:
    axes[0].set_title(title)

  if not violin_trials or not violin_labels:
    return
  labels = violin_labels
  for i, trial in enumerate(violin_trials):
    data = [np.array(label2array[la])[:, trial] for la in labels]
    quantile1, medians, quantile3 = [], [], []
    for d in data:
      q1, q2, q3 = np.percentile(d, [20, 50, 80])
      quantile1.append(q1)
      medians.append(q2)
      quantile3.append(q3)
    parts = axes[i + 1].violinplot(data, showmedians=False, showextrema=False)
    inds = np.arange(1, len(medians) + 1)
    axes[i + 1].scatter(
        inds, medians, marker='o', color='white', s=30, zorder=3)
    axes[i + 1].vlines(
        inds, quantile1, quantile3, color='k', linestyle='-', lw=5)
    for pc, la in zip(parts['bodies'], labels):
      pc.set_facecolor(colors[la])
      pc.set_edgecolor('black')
      pc.set_alpha(1)
    if 'x' in kwargs:
      axes[i + 1].set_title(f'{xlabel} = {kwargs["x"][trial]}')
    else:
      axes[i + 1].set_title(f'{xlabel} = {trial+1}')
    set_violin_axis_style(axes[i + 1], labels)
