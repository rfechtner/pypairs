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
  docker exec -it sl_test yum install -y python3 python3-setuptools
  docker exec -it sl_test easy_install pip
  docker exec -it sl_test git clone --depth=50 --branch=$TRAVIS_BRANCH https://github.com/rfechtner/pypairs.git rfechtner/pypairs
fi
