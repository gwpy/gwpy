#!/bin/bash
# Copyright (C) Duncan Macleod (2017-2019)
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

# get path to python and pip
PYTHON=$(which "python${PYTHON_VERSION}")
PIP="${PYTHON} -m pip"

# upgrade pip to understand python_requires
${PIP} install "pip>=9.0.0"

# upgrade setuptools to understand environment markers
${PIP} install "setuptools>=20.2.2" wheel

# install test dependencies
${PIP} install -r requirements-test.txt

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
