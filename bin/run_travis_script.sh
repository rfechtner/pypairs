#!/usr/bin/env bash

set -ev

if [ "${SYSTEM}" = "linux" ] || [ "${SYSTEM}" = "osx" ]; then
  pytest --cov=./
  if [ "${SYSTEM}" = "linux" ]; then
    rst2html.py --halt=2 README.rst >/dev/null
  fi
elif [ "${SYSTEM}" = "scientific" ]; then
  docker exec -it sl_test bash -c 'source /root/miniconda/bin/activate pypairs_test_env && cd /root/pypairs/ && pytest'
fi

export PRERELEASE=`echo $TRAVIS_TAG | grep -c "b"`
