# HyperBO - Prior Discovery
A Jax/Flax codebase for the algorithm in HyperBO described in *[Pre-trained Gaussian processes for Bayesian optimization](https://www.jmlr.org/papers/v25/23-0269.html)* published in the Journal of Machine Learning Research (JMLR).

**[PDF](https://arxiv.org/pdf/2109.08215.pdf)** | **[Blog post](https://ai.googleblog.com/2023/04/pre-trained-gaussian-processes-for.html)** | **[NeurIPS (Journal To Conference Track)](https://neurips.cc/virtual/2024/poster/98319) ** 

**[Colab Notebook](https://colab.research.google.com/github/google-research/hyperbo/blob/main/hyperbo/hyperbo_demo.ipynb)** | **[PD1 benchmark](https://github.com/google-research/hyperbo#pd1-benchmark)**

Disclaimer: This is not an officially supported Google product.

## Tutorial
Follow [HyperBO's Colab Notebook](https://colab.research.google.com/github/google-research/hyperbo/blob/main/hyperbo/hyperbo_demo.ipynb) or [Jupyter Notebook](https://github.com/google-research/hyperbo/blob/main/hyperbo/hyperbo_demo.ipynb).

Also see tests for a more comprehensive understanding of the usage.

## Installation
We recommend using Python 3.7 or 3.9 for stability.

To install the latest development version inside a virtual environment, run
```
python3 -m venv env-pd
source env-pd/bin/activate
pip install --upgrade pip
pip install "git+https://github.com/google-research/hyperbo.git#egg=hyperbo"
```

## PD1 benchmark
PD1 is a new hyperparameter tuning benchmark for optimizing deep learning models. To download the PD1 dataset, please copy and paste the following link to your browser's address bar.
```
http://storage.googleapis.com/gresearch/pint/pd1.tar.gz
```
See pd1/README.txt for more information. The data is licensed under the CC-BY 4.0 license.

If you'd like to use the evaluations at each training step, the relevant columns of the data frame are
```
'valid/ce_loss'
'train/ce_loss',
'train/error_rate',
```
etc. They will hold arrays aligned with the global_step column that indicates what training step the measurement was taken at.

See the "best_\*" columns for the best measurement achieved over training.


## GPax
[GPax](https://github.com/google-research/gpax) is a modular implementation of Gaussian processes used by HyperBO based on [Tensorflow Probability](https://www.tensorflow.org/probability) with Jax backend.

## Citation
Please cite our work if you would like to use the code.
```
@article{JMLR:v25:23-0269,
  author  = {Zi Wang and George E. Dahl and Kevin Swersky and Chansoo Lee and Zachary Nado and Justin Gilmer and Jasper Snoek and Zoubin Ghahramani},
  title   = {Pre-trained Gaussian Processes for Bayesian Optimization},
  journal = {Journal of Machine Learning Research},
  year    = {2024},
  volume  = {25},
  number  = {212},
  pages   = {1--83},
  url     = {http://jmlr.org/papers/v25/23-0269.html}
}
```
