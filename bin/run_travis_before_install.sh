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
  docker exec -it sl_test subscription-manager repos --enable rhel-7-server-optional-rpms --enable rhel-server-rhscl-7-rpms
  docker exec -it sl_test yum -y install @development
  docker exec -it sl_test yum -y install rh-python36
  docker exec -it sl_test yum -y install rh-python36-numpy rh-python36-scipy rh-python36-python-tools
  docker exec -it sl_test scl enable rh-python36 bash
  docker exec -it sl_test python3 -V
  docker exec -it sl_test git clone --depth=50 --branch=$TRAVIS_BRANCH https://github.com/rfechtner/pypairs.git rfechtner/pypairs
fi
