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
# Submit coverage data to coveralls.io
#

. ci/lib.sh

# fix paths in coverage file for docker-based runs
if [ ! -z ${DOCKER_IMAGE+x} ]; then
    sed -i 's|"'${GWPY_PATH}'|"'`pwd`'|g' .coverage;
fi

# install coveralls
if [[ "${TRAVIS_OS_NAME}" == "osx" ]]; then
    PYTHON=/opt/local/bin/python${PYTHON_VERSION}
    sudo -H ${PYTHON} -m pip install --quiet coveralls
else
    ${PIP} install --quiet coveralls
fi

# submit coverage results (unwrapping path to coveralls from python)
$(${PYTHON} -c "import sys; print(sys.prefix)")/bin/coveralls
