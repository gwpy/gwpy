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
sudo port -q install \
    gsed \
    ${PY_DIST} \
    ${PY_PREFIX}-setuptools \
    ${PY_PREFIX}-pip \
    ${PY_PREFIX}-jinja2 \
    ${PY_PREFIX}-gitpython

# make Portfile
cd ${GWPY_PATH}
GWPY_VERSION=$(python setup.py --version)
$PYTHON setup.py --quiet sdist
$PYTHON setup.py port --tarball dist/gwpy-${GWPY_VERSION}.tar.gz

# create mock portfile repo and install
PORT_REPO=$(pwd)/ports
GWPY_PORT_PATH=${PORT_REPO}/python/py-gwpy
mkdir -p ${GWPY_PORT_PATH}
cp Portfile ${GWPY_PORT_PATH}/

# munge Portfile for local install
gsed -i 's|pypi:g/gwpy|file://'$(pwd)'/dist/ \\\n                    pypi:g/gwpy|' ${GWPY_PORT_PATH}/Portfile

# add local port repo to sources
sudo gsed -i 's|rsync://rsync.macports|file://'${PORT_REPO}'\nrsync://rsync.macports|' /opt/local/etc/macports/sources.conf
cd ${PORT_REPO}
portindex
cd ${GWPY_PATH}

# set up utility to ping STDOUT every 10 seconds, prevents timeout
# https://github.com/travis-ci/travis-ci/issues/6591#issuecomment-275804717
set +x
write_visual_bells &  # <- prevent timeout
wvbpid=$!
disown
set -x

# install py-gwpy
# Note: we don't use the +gwf port, because ldas-tools-framecpp takes too
#       long to compile that the whole job times out in the end
sudo port -q install ${PY_PREFIX}-gwpy +nds2 +segments

# install extras (see requirements-dev.txt)
sudo port -q install \
    kerberos5 \
    ${PY_PREFIX}-lalsimulation \
    ${PY_PREFIX}-pymysql \
    ${PY_PREFIX}-sqlalchemy \
    ${PY_PREFIX}-psycopg2 \
    ${PY_PREFIX}-pandas \
    ${PY_PREFIX}-pytest \
    ${PY_PREFIX}-coverage \
    ${PY_PREFIX}-freezegun \
    ${PY_PREFIX}-sqlparse \
    ${PY_PREFIX}-beautifulsoup4

# install python2-only extras
if [ "${PY_MAJOR_VERSION}" -eq 2 ]; then
    sudo port -q install glue
fi

# install m2cyrpto if needed
if [ "$(port info --version dqsegdb)" == "version: 1.4.0" ]; then
    sudo port -q install ${PY_PREFIX}-m2crypto
fi

kill -9 $wvbpid &> /dev/null

# hacky fix for installing NOT mpl 2.1.x
#     this can be removed as soon as mpl 2.1.2 is released
MPL_VERSION=$(port -q installed ${PY_PREFIX}-matplotlib | grep active | \
              awk -F '[\@\_]' '{print $2}')
if [[ "${MPL_VERSION}" =~ 2.1.[01] ]]; then
    ${PIP} install "matplotlib >= 1.2.0, != 2.1.0, != 2.1.1"
fi
