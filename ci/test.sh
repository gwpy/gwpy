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

#
# Run the test suite for GWpy on the current system
#

. ci/lib.sh

# move to new empty directory (so that tests run on installed code)
mkdir -p test
pushd test

get_environment  # sets PIP variables etc
get_python_version  # sets PYTHON_VERSION

# macports PATH doesn't persist from install stage, which is annoying
if [[ ${TRAVIS_OS_NAME} == "osx" ]]; then
    . terryfy/travis_tools.sh
    export PATH=$MACPORTS_PREFIX/bin:$PATH
fi

set -ex

# upgrade pip to some minimal level to understand '.[tests]'
${PIP} install "pip>=7.0.0"

# install test dependencies
${PIP} install ${PIP_FLAGS} \
    "six>1.10.0" \
    "pytest>=3.1" \
    "pytest-cov" \
    "freezegun>0.2.3" \
    "sqlparse" \
    "bs4"
if [ "${PY_MAJOR_VERSION}" -eq 2 ]; then
    ${PIP} install mock
fi

# print installed packages
_gwpyloc=$(${PYTHON} -c 'import gwpy; print(gwpy.__file__)')
echo "------------------------------------------------------------------------"
echo
echo "GWpy installed to $_gwpyloc"
echo
echo "------------------------------------------------------------------------"

echo "Dependencies:"
echo "-------------"
${PYTHON} -m pip list installed --format=columns

# run tests
${PYTHON} -m pytest --pyargs gwpy --cov=gwpy

# deploy test results to coveralls
${PIP} install ${PIP_FLAGS} coveralls
coveralls || true

popd > /dev/null  # return to starting directory
