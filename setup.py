"""HyperBO.

See more details in the
[`README.md`](https://github.com/google-research/hyperbo).
"""

from setuptools import find_packages
from setuptools import setup

setup(
    name='hyperbo',
    version='0.0.1',
    description='hyperbo',
    author='HyperBO Team',
    author_email='wangzi@google.com',
    url='http://github.com/google-research/hyperbo',
    license='Apache 2.0',
    packages=find_packages(),
    install_requires=[
        'absl-py>=0.8.1',
        'clu',
        'flax',
        'jax',
        'ml_collections',
        'numpy>=1.7',
        'optax',
        'pandas',
        'tensorflow_probability',
        'tensorflow==2.5.0',
    ],
    extras_require={},
    classifiers=[
        # https://pypi.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='bayesian optimization pre-training',
)
