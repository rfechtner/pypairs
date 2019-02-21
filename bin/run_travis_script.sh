#!/usr/bin/env bash

set -ev

if [ "${SYSTEM}" = "linux" ] || [ "${SYSTEM}" = "osx" ]; then

  pytest --cov=./
  
  if [ "${SYSTEM}" = "linux" ]; then 
  
    rst2html.py --halt=2 README.rst >/dev/null
    export PRERELEASE=`echo $TRAVIS_TAG | grep -c "b"`
    
  fi
fi
