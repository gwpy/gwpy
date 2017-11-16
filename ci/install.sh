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
# Build system packages
#

cd ${GWPY_PATH}

set -e

# upgrade pip
${PIP} install --upgrade pip

# get version number for install scripts to use
GWPY_VERSION=`${PYTHON} setup.py version | grep Version | cut -d\  -f2`

# install for this OS
if [ -z ${DOCKER_IMAGE} ]; then  # simple
    ${PIP} install .
elif [[ ${DOCKER_IMAGE} =~ el[0-9]+$ ]]; then  # SLX
    . ci/install-el.sh
else
    . ci/install-debian.sh
fi

# install extras
${PIP} install -r requirements-dev.txt

echo "------------------------------------------------------------------------"
echo
echo "GWpy installed to `${PYTHON} -c 'import gwpy; print(gwpy.__file__)'`"
echo
echo "------------------------------------------------------------------------"

set +e
