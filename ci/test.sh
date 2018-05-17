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

cd ${GWPY_PATH}

. ci/lib.sh

set -ex && trap 'set +xe' RETURN

# macports PATH doesn't persist from install stage, which is annoying
if [ $(get_package_manager) == port ]; then
    . terryfy/travis_tools.sh
    export PATH=$MACPORTS_PREFIX/bin:$PATH
fi

get_environment  # sets PIP variables etc
get_python_version  # sets PYTHON_VERSION

# install test dependencies
${PIP} install ${PIP_FLAGS} coverage "setuptools>=17.1" "pytest>=3.1"
COVERAGE=coverage-${PYTHON_VERSION}

# fix broken glue dependency
#     this is required because the LIGO glue package isn't
#     distributed as lscsoft-glue in system packages,
#     only in pypi, so once it is, this line should have
#     no effect
${PIP} install ${PIP_FLAGS} lscsoft-glue

# run tests
${COVERAGE} run ./setup.py test --addopts gwpy/tests/
