#!/usr/bin/env bash

set -ev

if [ "${SYSTEM}" = "linux" ] || [ "${SYSTEM}" = "osx" ]; then
  ./bin/install_conda.sh
  rm -rf ./cache/ || true
  mkdir ./cache/
  rm -rf ./write/ || true
  mkdir ./write/
  rm -rf ./figures/ || true
  mkdir ./figures/
elif [ "${SYSTEM}" = "scientific" ]; then
  docker pull scientificlinux/sl
  docker run -d --name sl_test scientificlinux/sl:7
  docker exec sl_test sudo yum -y update
  docker exec sl_test sudo yum install -y python36 python36-setuptools
  docker exec sl_test sudo easy_install-3.6 pip
fi
