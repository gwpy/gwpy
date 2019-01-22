#!/bin/bash
# Copyright (C) Duncan Macleod (2017)
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
# Run the test suite for GWpy on the current system
#

# get path to python and pip
PYTHON="python${PYTHON_VERSION:-${TRAVIS_PYTHON_VERSION}}"
PYTHON_PREFIX=$(${PYTHON} -c "import sys; print(sys.prefix)")

# install with sudo on macports
if [[ "${PYTHON_PREFIX}" =~ "/opt/local/"* ]]; then
    PIP="sudo -H ${PYTHON} -m pip"
else
    PIP="${PYTHON} -m pip"
fi

# upgrade setuptools in order to understand environment markers
${PIP} install "pip>=8.0.0" "setuptools>=20.2.2"

# install test dependencies
${PIP} install ${PIP_FLAGS} -r requirements-test.txt

# try and install pytest-cov again, which forces pip to make sure
# that the right version of pytest is installed
if grep -q "pytest-cov" requirements-test.txt; then
    ${PIP} install ${PIP_FLAGS} pytest-cov
fi

# list all packages
${PIP} list installed

# run tests with coverage
${PYTHON} -m pytest --pyargs gwpy --cov=gwpy
