#!/usr/bin/env bash
set -ev
echo "Running before install"
if [ "${SYSTEM}" = "linux" ] || [ "${SYSTEM}" = "osx" ]; then
  echo "Installing Conda"
  ./bin/install_conda.sh
elif [ "${SYSTEM}" = "scientific" ]; then
  echo "Setting up scientifix linux docker container"
  docker pull scientificlinux/sl
  docker run -it -d --name sl_test scientificlinux/sl:7
  docker ps -a
  docker exec -it sl_test cat /etc/os-release
  docker exec -it sl_test yum -y update
  docker exec -it sl_test yum install wget
  docker exec -it sl_test bash -c 'mkdir -p "$HOME"/download/'
  docker exec -it sl_test bash -c 'wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$HOME"/download/miniconda.sh'
  docker exec -it sl_test bash -c '"$HOME"/download/miniconda.sh -b -p "$HOME"/miniconda'
  docker exec -it sl_test python -V
  docker exec -it sl_test git clone --depth=50 --branch=$TRAVIS_BRANCH https://github.com/rfechtner/pypairs.git rfechtner/pypairs
fi
