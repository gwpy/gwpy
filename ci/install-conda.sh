#!/bin/bash
# Copyright (C) Duncan Macleod (2018-2020)
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

# get local functions
. ci/lib.sh

#
# Install GWpy and dependencies using Conda
#

install_miniconda
conda_init

# configure miniconda
${CONDA_EXE} config --set always_yes yes --set changeps1 no
${CONDA_EXE} config --add channels conda-forge
${CONDA_EXE} update --quiet --yes conda
${CONDA_EXE} info --all

# create environment for tests
${CONDA_EXE} create --name "${GWPY_CONDA_ENV_NAME}" --quiet --yes python=${PYTHON_VERSION} pip setuptools

# install conda dependencies (based on pip requirements file)
${CONDA_EXE} run --name "${GWPY_CONDA_ENV_NAME}" \
python ./ci/parse-conda-requirements.py requirements-dev.txt -o conda-reqs.txt
${CONDA_EXE} install --name "${GWPY_CONDA_ENV_NAME}" --quiet --yes --file conda-reqs.txt --update-all
rm -f conda-reqs.txt  # clean up

# -- install other conda packages that aren't represented in the requirements file
# all platforms
${CONDA_EXE} install --name "${GWPY_CONDA_ENV_NAME}" --quiet --yes \
    "python-framel>=8.40.1" \
    "python-nds2-client" \
;
# unix
if ! on_windows; then
    ${CONDA_EXE} install --name "${GWPY_CONDA_ENV_NAME}" --quiet --yes \
        "python-lal" \
        "python-lalframe" \
        "python-lalsimulation" \
        "python-ldas-tools-framecpp" \
    ;
fi
