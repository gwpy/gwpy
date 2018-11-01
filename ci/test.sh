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

PYTHON="python${PYTHON_VERSION:-${TRAVIS_PYTHON_VERSION}}"

# upgrade setuptools in order to understand environment markers
${PYTHON} -m pip install "pip>=8.0.0" "setuptools>=20.2.2"

# install test dependencies
${PYTHON} -m pip install ${PIP_FLAGS} -r requirements-test.txt

# run tests with coverage
${PYTHON} -m pytest --pyargs gwpy --cov=gwpy
