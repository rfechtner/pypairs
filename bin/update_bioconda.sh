#!/usr/bin/env bash

set -ev

# Install biconda-utils
if ! type bioconda-utils > /dev/null; then
  conda update --yes conda
  conda config --add channels defaults
  conda config --add channels conda-forge
  conda config --add channels bioconda
  git clone https://github.com/bioconda/bioconda-utils
  cd bioconda-utils
  python setup.py install
  cd .. 
fi

# Prepare update of pypairs package
git clone https://github.com/rfechtner/bioconda-recipes.git
cd bioconda-recipes
git checkout -b update-pypairs

# Get latest version & sha256 from pypairs
export STRICT_VERSION=`curl https://pypi.org/pypi/pypairs/json | python -c "import sys, json; print(str(json.load(sys.stdin)['info']['version']))"`
export SHA256=`curl https://pypi.org/pypi/pypairs/json | python -c "import sys, json; print(str(json.load(sys.stdin)['urls'][0]['digests']['sha256']))"`

# Update meta.yaml
cd recipes/pypairs
sed -i '2s/.*/{% set version = "'$STRICT_VERSION'" %}/' meta.yaml
sed -i '10s/.*/  sha256: '$SHA256'/' meta.yaml
cd ../..

# Use bioconda-utils to make pr
bioconda-utils update recipes/ config.yml --packages pypairs --create-pr
