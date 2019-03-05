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

PYTHON_VERSION=$(echo "${PYTHON_VERSION:-${TRAVIS_PYTHON_VERSION}}" | cut -d. -f-2)


if ! which conda 1> /dev/null; then
    if test ! -f ${HOME}/miniconda/etc/profile.d/conda.sh; then
        # install conda
        [ "$(uname)" == "Darwin" ] && MC_OSNAME="MacOSX" || MC_OSNAME="Linux"
        MINICONDA="Miniconda${PYTHON_VERSION%%.*}-latest-${MC_OSNAME}-x86_64.sh"
        curl https://repo.continuum.io/miniconda/${MINICONDA} -o miniconda.sh
        bash miniconda.sh -b -u -p ${HOME}/miniconda
    fi
    source ${HOME}/miniconda/etc/profile.d/conda.sh
    set -ex  # gets unset by source
fi
hash -r

# get CONDA base path
CONDA_PATH=$(conda info --base)

# configure
conda config --set always_yes yes
conda config --add channels conda-forge

# update conda
conda update --quiet conda

conda info --all

# create environment for tests (if needed)
if [ ! -f ${CONDA_PATH}/envs/gwpyci/conda-meta/history ]; then
    conda create --name gwpyci python=${PYTHON_VERSION} numpy gwpy
fi
conda activate gwpyci || source activate gwpyci
PYTHON=$(which python)

# install conda dependencies (based on pip requirements file)
${PYTHON} ./ci/parse-conda-requirements.py requirements-dev.txt -o conda-reqs.txt
conda install --name gwpyci --quiet --yes --file conda-reqs.txt
rm -f conda-reqs.txt  # clean up

# install other conda packages that aren't represented in the requirements file
conda install --name gwpyci --quiet --yes \
    python-lal \
    python-lalframe \
    python-lalsimulation \
    python-ldas-tools-framecpp \
    python-nds2-client \
    root_numpy

# install gwpy into this environment
${PYTHON} -m pip install ${PIP_FLAGS} . --ignore-installed --no-deps
