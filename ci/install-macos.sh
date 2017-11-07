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

# get python information
get_environment

sudo port install gsed

# make Portfile
cd ${GWPY_PATH}
python setup.py port

# create mock portfile repo and install
PORT_REPO=`pwd`/ports
mkdir -p ${PORT_REPO}/science/gwpy
cp Portfile ${PORT_REPO}/science/gwpy

# munge Portfile for local install
gsed -i 's|'\
    'pypi:g/gwpy|'\
    'file://'`pwd`'/dist/ \\\n                    pypi:g/gwpy|' \
    ${PORT_REPO}/science/gwpy/Portfile

# add local port repo to sources
sudo gsed -i 's|'\
    'rsync://rsync.macports|'\
    'file://'${PORT_REPO}'\nrsync://rsync.macports|' \
    /opt/local/etc/macports/sources.conf
cd ${PORT_REPO}
portindex

# install port
port install ${PY_PREFIX}-gwpy +gwf +nds2 +hdf5
