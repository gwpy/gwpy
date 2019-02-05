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

set -x
trap 'set +ex' RETURN

function write_visual_bells() {
    while true; do
        echo -en "\a"
        sleep 10
    done
}

#
# Build Port for GWpy
#

git clone --quiet --recursive https://github.com/MacPython/terryfy.git
set +x
. terryfy/travis_tools.sh
set -ex  # travis_tools.sh sets +x on its way out

export COLUMNS=80  # https://github.com/travis-ci/travis-ci/issues/5407
install_macports

PYTHON_VERSION=${PYTHON_VERSION:-${TRAVIS_PYTHON_VERSION}}
PY_DIST="python${PYTHON_VERSION/./}"
PY_PREFIX="py${PYTHON_VERSION/./}"

# install basic ports we need
sudo port -q install \
    gsed \
    ${PY_DIST} \
    ${PY_PREFIX}-setuptools \
    ${PY_PREFIX}-pip \
    ${PY_PREFIX}-jinja2 \
    ${PY_PREFIX}-gitpython
PYTHON=$(which "python${PYTHON_VERSION}")

# make Portfile
GWPY_VERSION=$($PYTHON setup.py --version)
$PYTHON setup.py --quiet sdist
$PYTHON setup.py port --tarball dist/gwpy-${GWPY_VERSION}.tar.gz

# create mock portfile repo and install
PORT_REPO="$(pwd)/ports"
GWPY_PORT_PATH="${PORT_REPO}/python/py-gwpy"
mkdir -p ${GWPY_PORT_PATH}
cp Portfile ${GWPY_PORT_PATH}/

# munge Portfile for local install
gsed -i 's|pypi:g/gwpy|file://'$(pwd)'/dist/ \\\n                    pypi:g/gwpy|' ${GWPY_PORT_PATH}/Portfile

# add local port repo to sources
(
cd ${PORT_REPO}
sudo gsed -i 's|rsync://rsync.macports|file://'$(pwd)'\nrsync://rsync.macports|' /opt/local/etc/macports/sources.conf
portindex
)

# set up utility to ping STDOUT every 10 seconds, prevents timeout
# https://github.com/travis-ci/travis-ci/issues/6591#issuecomment-275804717
set +x
write_visual_bells &  # <- prevent timeout
wvbpid=$!
disown
set -x

# install scipy with openblas to fix symbols error
sudo port -q install ${PY_PREFIX}-scipy +openblas

# install py-gwpy
sudo port -q install ${PY_PREFIX}-gwpy

# install testing dependencies
sudo port -q install \
    ${PY_PREFIX}-pytest \
    ${PY_PREFIX}-pytest-cov \
    ${PY_PREFIX}-freezegun \
    ${PY_PREFIX}-sqlparse \
    ${PY_PREFIX}-beautifulsoup4
if [[ "${PY_PREFIX}" == "py27" ]]; then
    sudo port -q install py27-mock
fi

# install extras (see requirements-dev.txt)
#sudo port -q install \
#    kerberos5 \
#    ${PY_PREFIX}-gwdatafind \
#    ${PY_PREFIX}-lal \
#    ${PY_PREFIX}-lalframe \
#    ${PY_PREFIX}-lalsimulation \
#    ${PY_PREFIX}-nds2-client \
#    ${PY_PREFIX}-pymysql \
#    ${PY_PREFIX}-sqlalchemy \
#    ${PY_PREFIX}-psycopg2 \
#    ${PY_PREFIX}-pandas
#
## install python2.7-only packages
#if [[ "${PY_PREFIX}" == "py27" ]] then
#    sudo port -q install dqsegdb
#    # install m2crypto if needed
#    if [ "$(port info --version dqsegdb)" == "version: 1.4.0" ]; then
#        sudo port -q install ${PY_PREFIX}-m2crypto
#    fi
#fi

# kill the fancy timeout thing
kill -9 $wvbpid &> /dev/null
