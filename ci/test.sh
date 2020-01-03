#!/bin/bash
# Copyright (C) Duncan Macleod (2017-2020)
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
unset -f cd
unset -f pushd
unset -f popd

#
# Run the test suite for GWpy on the current system
#

# reactivate environmennt
if [ -n ${CIRCLECI} ] && [ -d /opt/conda/envs ]; then
    conda activate gwpyci || { source activate gwpyci; set -ex; }
fi

# get path to python and pip
PYTHON_VERSION=$(echo "${PYTHON_VERSION:-${TRAVIS_PYTHON_VERSION}}" | cut -d. -f-2)
PYTHON=$(which "python${PYTHON_VERSION}")
PIP="${PYTHON} -m pip"

# upgrade setuptools in order to understand environment markers
${PIP} install ${PIP_FLAGS} "pip>=8.0.0" "setuptools>=20.2.2"

# install test dependencies
${PIP} install ${PIP_FLAGS} -r requirements-test.txt

# try and install pytest-cov again, which forces pip to make sure
# that the right version of pytest is installed
if grep -q "pytest-cov" requirements-test.txt; then
    ${PIP} install ${PIP_FLAGS} pytest-cov
fi

# list all packages
if [ -z ${CONDA_PREFIX} ]; then
    ${PIP} list installed
else
    conda list --name gwpyci
fi

# run tests with coverage - we use a separate directory here
# to guarantee that we run the tests from the installed code
mkdir -p tests
pushd tests

# run standard test suite
${PYTHON} -m pytest \
    --pyargs gwpy \
    --cov gwpy \
    --cov-report xml:coverage.xml \
    --junitxml junit1.xml \
    --numprocesses 2

# run examples test suite
${PYTHON} -m pytest \
    ../examples/test_examples.py \
    --verbose \
    --cov gwpy \
    --cov-append \
    --cov-report xml:coverage.xml \
    --junitxml junit2.xml || {
# handle exit code 5 (all tests skipped) as pass
EC_="$?";
[ "${EC_}" -ne 5 ] && exit "${EC_}";
}

# combine junit files from each pytest instance
${PIP} install junitparser
${PYTHON} -c "
from junitparser import JUnitXml
xml1 = JUnitXml.fromfile('junit1.xml')
xml1 += JUnitXml.fromfile('junit2.xml')
xml1.write('junit.xml')"
rm -f junit1.xml junit2.xml

popd
