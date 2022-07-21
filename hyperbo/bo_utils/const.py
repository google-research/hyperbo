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

"""Definition of constants."""
from hyperbo.bo_utils import acfun
from hyperbo.bo_utils import data
from hyperbo.gp_utils import kernel
from hyperbo.gp_utils import mean

MEAN = {
    'constant': mean.constant,
    'linear': mean.linear,
    'linear_mlp': mean.linear_mlp,
}

KERNEL = {
    'squared_exponential': kernel.squared_exponential,
    'matern32': kernel.matern32,
    'matern52': kernel.matern52,
    'dot_product': kernel.dot_product,
    'dot_product_mlp': kernel.dot_product_mlp,
}

ACFUN = {
    'expected_improvement': acfun.expected_improvement,
    'probability_of_improvement': acfun.probability_of_improvement,
    'ucb3': acfun.ucb3,
    'random_search': acfun.random_search,
}

ACFUN_SUB = {
    'expected_improvement': acfun.expected_improvement_sub,
    'probability_of_improvement': acfun.probability_of_improvement_sub,
    'ucb': acfun.ucb_sub,
}

EPS = 1e-6

HYPERBO_DATASETS = {
    # 'grid2020': data.grid2020,
    'pd1': data.pd1,
    # 'pd2': data.pd2,
    'random': data.random,
}

INPUT_SAMPLERS = {}

# For offline experiment manage
RAND = 'rand'
STBO = 'stbo'
MTBO = 'mtbo'
STBOV = 'gp'
HBO = 'hyperbo'
HBO_SS = 'hyperbo_ss'
HBO_NLL = 'hyperbo_nll'
HBO_NLLKL = 'hyperbo_nllkl'
HBO_NLLEUC = 'hyperbo_nlleuc'

CONTEXTUAL_METHODS = ['rfgp', 'mimo', STBOV]

HBO_METHODS = [HBO_SS, HBO_NLL, HBO_NLLKL, HBO_NLLEUC]
OFFLINE_METHODS = [RAND, STBO, MTBO, HBO, HBO_SS] + CONTEXTUAL_METHODS

ONLINE_METHODS = [STBO, MTBO] + HBO_METHODS
USE_HGP = [HBO_SS]
ST_METHODS = [STBO, STBOV]
