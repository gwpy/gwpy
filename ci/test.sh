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

cd ${GWPY_PATH}

. ci/lib.sh

set -x
set -e

get_environment

# install test dependencies
${PIP} install coverage "pytest>=3.1"

# fix broken glue dependency
#     this is required because the LIGO glue package isn't
#     distributed as lscsoft-glue in system packages,
#     only in pypi, so once it is, this line should have
#     no effect
pip install lscsoft-glue

# run tests
coverage run ./setup.py test --addopts "gwpy/tests/ ${TEST_FLAGS}"

set +e
set +x
