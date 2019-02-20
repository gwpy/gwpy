#!/bin/bash
# Copyright (C) Duncan Macleod (2018-2019)
#
# This file is part of GWpy.
#
# GWpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GWpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>.

set -ex
trap 'set +ex' RETURN

#
# Install GWpy and dependencies using Conda
#

PYTHON_VERSION=${PYTHON_VERSION:-${TRAVIS_PYTHON_VERSION}}

# install conda
if [[ "${TRAVIS_OS_NAME}" == "osx" ]]; then
    MINICONDA="Miniconda3-latest-MacOSX-x86_64.sh"
else
    MINICONDA="Miniconda3-latest-Linux-x86_64.sh"
fi
curl https://repo.continuum.io/miniconda/${MINICONDA} -o miniconda.sh
bash miniconda.sh -b -u -p ${HOME}/miniconda
export PATH="${HOME}/miniconda/bin:${PATH}"
hash -r
conda config --set always_yes yes --set changeps1 no
conda config --add channels conda-forge
conda update --quiet conda
conda info --all

# install correct versino of python, and gwpy's dependencies only
conda install --only-deps python=${PYTHON_VERSION} gwpy

# install conda dependencies (based on pip requirements file)
python ./ci/parse-conda-requirements.py requirements-dev.txt -o conda-reqs.txt
conda install --quiet --yes --file conda-reqs.txt
rm -f conda-reqs.txt  # clean up

# install other conda packages that aren't represented in the requirements file
conda install --quiet --yes \
    python-lal \
    python-lalframe \
    python-lalsimulation \
    python-nds2-client

# install gwpy into this environment
python -m pip install .
