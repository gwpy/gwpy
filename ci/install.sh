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

. ci/lib.sh

set -x && trap 'set +x' RETURN

get_environment

# install for this OS
if [[ ${TRAVIS_OS_NAME} == "osx" ]]; then  # macports
    . ci/install-macos.sh
elif [[ ${DOCKER_IMAGE} =~ :el[0-9]+$ ]]; then  # SLX
    . ci/install-el.sh
elif [ -n "${DOCKER_IMAGE}" ]; then  # debian
    . ci/install-debian.sh
else  # simple pip build
    [[ ${TRAVIS_PYTHON_VERSION} == "nightly" ]] && \
        ${PIP} install Cython ${PIP_FLAGS} --install-option="--no-cython-compile"
    ${PIP} install . ${PIP_FLAGS}
fi

# install python extras
${PIP} install --quiet ${PIP_FLAGS} "setuptools>=25"
${PIP} install -r requirements-dev.txt --quiet ${PIP_FLAGS}

cd /tmp
_gwpyloc=`${PYTHON} -c 'import gwpy; print(gwpy.__file__)'`
echo "------------------------------------------------------------------------"
echo
echo "GWpy installed to $_gwpyloc"
echo
echo "------------------------------------------------------------------------"
cd - 1> /dev/null
