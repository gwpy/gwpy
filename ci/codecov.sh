#!/bin/bash
# Copyright (C) Duncan Macleod (2019)
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
# Submit coverage data to codecov.io
#

# reactivate environmennt
if [ -n ${CIRCLECI} ] && [ -d /opt/conda/envs ]; then
    conda activate gwpyci || source activate gwpyci
fi

# get path to python
PYTHON_VERSION=$(echo "${PYTHON_VERSION:-${TRAVIS_PYTHON_VERSION}}" | cut -d. -f-2)
PYTHON=$(which "python${PYTHON_VERSION}")

# install codecov
${PYTHON} -m pip install ${PIP_FLAGS} coverage codecov

# find job name
_JOBNAME=${CIRCLE_JOB:-${TRAVIS_JOB_NAME}}

# submit coverage results
${PYTHON} -m codecov --file tests/coverage.xml --flags $(uname) python${PYTHON_VERSION/./} ${_JOBNAME%%:*}
