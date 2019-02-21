#!/usr/bin/env bash
set -ev
echo "Running before install"
if [ "${SYSTEM}" = "linux" ] || [ "${SYSTEM}" = "osx" ]; then
  echo "Installing Conda"
  ./bin/install_conda.sh
elif [ "${SYSTEM}" = "scientific" ]; then
  # Setting up scientifix linux docker container
  docker pull scientificlinux/sl
  docker run -it -d --name sl_test --privileged=true scientificlinux/sl:7
  docker ps -a
  # Verify os
  docker exec -it sl_test cat /etc/os-release
  # Install miniconda
  docker exec -it sl_test yum -y update
  docker exec -it sl_test yum -y install wget
  docker exec -it sl_test yum -y install gcc
  docker exec -it sl_test bash -c 'mkdir -p /root/download/'
  docker exec -it sl_test bash -c 'wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/download/miniconda.sh'
  docker exec -it sl_test bash -c 'chmod +x /root/download/miniconda.sh'
  docker exec -it sl_test bash -c '/root/download/miniconda.sh -b -p /root/miniconda'
  docker exec -it sl_test bash -c '/root/miniconda/bin/conda create -y -n pypairs_test_env python=3.6'
  docker exec -it sl_test bash -c 'source /root/miniconda/bin/activate pypairs_test_env && python -V'
  docker exec -it sl_test bash -c 'source /root/miniconda/bin/activate pypairs_test_env && pip install pytest pytest-cov'
  # Copy repo
  docker cp ./. sl_test:/root/pypairs/
  docker exec -it sl_test cd /root/pypairs/
fi
