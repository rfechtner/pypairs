#!/usr/bin/env bash

set -ev

if [ "${SYSTEM}" = "linux" ] || [ "${SYSTEM}" = "osx" ]; then
  pytest --cov=./
  rst2html.py --halt=2 README.rst >/dev/null
  export PRERELEASE=`echo $TRAVIS_TAG | grep -c "b"`
elif [ "${SYSTEM}" = "scientific" ]; then
  docker exec sl_test python -m pytest
fi
