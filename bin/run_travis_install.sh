#!/usr/bin/env bash

set -ev

if [ "${SYSTEM}" = "linux" ] || [ "${SYSTEM}" = "osx" ]; then
  source activate pypairs_test_env
  pip install -y Sphinx pytest pytest-cov docutils sphinx-autodoc-typehints sphinx-rtd-theme codecov
  pip install -r requirements
  pip install -e .[plotting]
elif [ "${SYSTEM}" = "scientific" ]; then
  docker exec -it sl_test bash -c 'source /root/miniconda/bin/activate pypairs_test_env && cd /root/pypairs/ && pip install -r requirements.txt'
  docker exec -it sl_test bash -c 'source /root/miniconda/bin/activate pypairs_test_env && cd /root/pypairs/ && pip install -e .[plotting]'
fi
