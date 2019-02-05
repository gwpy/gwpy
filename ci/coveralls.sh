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
# Submit coverage data to coveralls.io
#

PYTHON="python${PYTHON_VERSION:-${TRAVIS_PYTHON_VERSION}}"
PYTHON_PREFIX=$(${PYTHON} -c "import sys; print(sys.prefix)")

# install coveralls (using sudo on macports)
if [[ "${PYTHON_PREFIX}" =~ "/opt/local/"* ]]; then
    PIP="sudo -H ${PYTHON} -m pip"
else
    PIP="${PYTHON} -m pip"
fi
${PIP} install --quiet coveralls

# submit coverage results
${PYTHON} -m coveralls
