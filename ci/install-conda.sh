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

#
# Install GWpy and dependencies using Conda
#

# install miniconda
if ! which conda 1> /dev/null; then
    if test ! -f ${HOME}/miniconda/etc/profile.d/conda.sh; then
        # install conda
        [ "$(uname)" == "Darwin" ] && MC_OSNAME="MacOSX" || MC_OSNAME="Linux"
        MINICONDA="Miniconda${PYTHON_VERSION%%.*}-latest-${MC_OSNAME}-x86_64.sh"
        curl -L https://repo.continuum.io/miniconda/${MINICONDA} -o miniconda.sh
        bash miniconda.sh -b -u -p ${HOME}/miniconda
    fi
    source ${HOME}/miniconda/etc/profile.d/conda.sh
fi
hash -r

# get CONDA base path
CONDA_PATH=$(conda info --base)

# configure miniconda
conda config --set always_yes yes --set changeps1 no
conda config --add channels conda-forge
conda update --quiet --yes conda
conda info --all

# create environment for tests (if needed)
if [ ! -f ${CONDA_PATH}/envs/gwpyci/conda-meta/history ]; then
    conda create --name gwpyci python=${PYTHON_VERSION} pip setuptools
fi

# install conda dependencies (based on pip requirements file)
conda run --name gwpyci \
python ./ci/parse-conda-requirements.py requirements-dev.txt -o conda-reqs.txt
conda install --name gwpyci --quiet --yes --file conda-reqs.txt --update-all
rm -f conda-reqs.txt  # clean up

# install other conda packages that aren't represented in the requirements file
conda install --name gwpyci --quiet --yes \
    "python-framel>=8.40.1" \
    "python-lal" \
    "python-lalframe" \
    "python-lalsimulation" \
    "python-ldas-tools-framecpp" \
    "python-nds2-client" \
    "root>=6.20" \
    "root_numpy" \
;

# activate the environment
. ${CONDA_PATH}/etc/profile.d/conda.sh
conda activate gwpyci
