#!/usr/bin/env bash

set -ev

if [ "${SYSTEM}" = "linux" ] || [ "${SYSTEM}" = "osx" ]; then
  pip freeze
  pytest --cov=./
  rst2html.py --halt=2 README.rst >/dev/null
  export PRERELEASE=`echo $TRAVIS_TAG | grep -c "b"`
elif [ "${SYSTEM}" = "scientific" ]; then
  docker exec -it sl_test bash -c 'source /root/miniconda/bin/activate pypairs_test_env && cd /root/pypairs/ && pytest'
fi
