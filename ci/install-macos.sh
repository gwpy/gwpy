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
# Build Port for GWpy
#

git clone https://github.com/MacPython/terryfy.git
. terryfy/travis_tools.sh
set -x  # travis_tools.sh sets +x on its way out

export COLUMNS=80  # https://github.com/travis-ci/travis-ci/issues/5407
install_macports

# get python information
get_environment

# install basic ports we need
sudo port install gsed ${PY_DIST} ${PR_PREFIX}-setuptools ${PY_PREFIX}-pip

# make Portfile
cd ${GWPY_PATH}
python setup.py port

# create mock portfile repo and install
PORT_REPO=`pwd`/ports
GWPY_PORT_PATH=${PORT_REPO}/python/py-gwpy
mkdir -p ${GWPY_PORT_PATH}
cp Portfile ${GWPY_PORT_PATH}/

# munge Portfile for local install
gsed -i 's|pypi:g/gwpy|file://'`pwd`'/dist/ \\\n                    pypi:g/gwpy|' ${GWPY_PORT_PATH}/Portfile

# add local port repo to sources
sudo gsed -i 's|rsync://rsync.macports|file://'${PORT_REPO}'\nrsync://rsync.macports|' /opt/local/etc/macports/sources.conf
cd ${PORT_REPO}
portindex

# install port (install +gwf separately because framecpp takes forever)
sudo port -N install ${PY_PREFIX}-gwpy +nds2 +hdf5
sudo port -N install ${PY_PREFIX}-gwpy +gwf
