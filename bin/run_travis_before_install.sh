#!/usr/bin/env bash
set -ev
if [ "${SYSTEM}" = "scientific" ]; then
  echo "Setting up scientifix linux docker container"
  docker pull scientificlinux/sl
  docker run -d -p 127.0.0.1:80:80 --name sl_test scientificlinux/sl:7
  docker ps | grep -q sl_test
  cat /etc/os-release
  docker exec sl_test cat /etc/os-release
  docker exec sl_test sudo yum -y update
  docker exec sl_test sudo yum install -y python36 python36-setuptools
  docker exec sl_test sudo easy_install-3.6 pip
fi
