# HyperBO - Prior Discovery
A Jax/Flax codebase for prior discovery in meta Bayesian optimization.
The algorithm and analyses can be found in *[Pre-training helps Bayesian optimization too](https://arxiv.org/abs/2109.08215)*.

Disclaimer: This is not an officially supported Google product.

## Installation
We recommend using Python 3.7 for stability.

To install the latest development version inside a virtual environment, run
```
python3 -m venv env-pd
source env-pd/bin/activate
pip install --upgrade pip
pip install "git+https://github.com/google-research/hyperbo.git#egg=hyperbo"
```

## Dataset
Please [download the dataset](http://storage.googleapis.com/gresearch/pint/pd1.tar.gz) and see pd1/README.txt for more information.

## Usage
See tests.

## Citing
```
@article{wang2021hyperbo,
  title={Pre-training helps Bayesian optimization too},
  author={Wang, Zi and Dahl, George E and Swersky, Kevin and Lee, Chansoo and Mariet, Zelda and Nado, Zachary and Gilmer, Justin and Snoek, Jasper and Ghahramani, Zoubin},
  journal={arXiv preprint arXiv:2109.08215},
  year={2022}
}
```
