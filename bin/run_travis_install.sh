#!/usr/bin/env bash

set -ev

if [ "${SYSTEM}" = "linux" ] || [ "${SYSTEM}" = "osx" ]; then
  source $HOME/miniconda/bin/activate
  conda install python=3.6 numpy numba pandas scikit-learn colorama Sphinx pytest pytest-cov docutils pytables -y
  pip install sphinx-autodoc-typehints sphinx-rtd-theme codecov
  pip install -e .[plotting]
fi
