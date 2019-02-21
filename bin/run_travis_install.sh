#!/usr/bin/env bash

set -ev

if [ "${SYSTEM}" = "linux" ] || [ "${SYSTEM}" = "osx" ]; then
  source $HOME/miniconda/bin/activate
  conda install python=3.6 numpy numba pandas scikit-learn colorama Sphinx pytest pytest-cov docutils pytables -y
  pip install sphinx-autodoc-typehints sphinx-rtd-theme codecov pytest
  pip install -e .[plotting]
elif [ "${SYSTEM}" = "scientific" ]; then
  docker exec sl_test pip install -m Sphinx pytest pytest-cov docutils sphinx-autodoc-typehints sphinx-rtd-theme codecov
  docker exec sl_test -e .[plotting]
fi
