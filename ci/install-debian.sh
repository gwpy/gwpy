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
# Build Debian package
#


if [ -z $PYTHON ]; then  # correct python version not installed
    apt-get install -yqq ${PY_DIST}
    PYTHON=`which python${PYTHON_VERSION}`
fi

# install build dependencies
apt-get update -yqq
apt-get install -yqq \
    git \
    debhelper \
    ${PY_PREFIX}-all \
    ${PY_PREFIX}-setuptools \
    ${PY_PREFIX}-pip

if [ ${PY_PREFIX} == "python" ]; then
    apt-get install -yqq python-git
else
    $PIP install GitPython
fi

$PIP install "setuptools>33"

# get versions
GWPY_VERSION=`$PYTHON setup.py version | grep Version | cut -d\  -f2`
GWPY_RELEASE=${GWPY_VERSION%%+*}

# prepare the tarball
$PYTHON changelog.py -f deb -s "v0.5" > debian/changelog
$PYTHON setup.py sdist

# make the debian package
mkdir -p dist/debian
pushd dist/debian
cp ../gwpy-${GWPY_VERSION}.tar.gz ../gwpy_${GWPY_RELEASE}.orig.tar.gz
tar -xf ../gwpy_${GWPY_RELEASE}.orig.tar.gz --strip-components=1
dpkg-buildpackage -us -uc
popd

# print and install the deb
GWPY_DEB="dist/${PY_PREFIX}-gwpy_${GWPY_RELEASE}-1_all.deb"
echo "-------------------------------------------------------"
dpkg --info ${GWPY_DEB}
echo "-------------------------------------------------------"
dpkg --install ${GWPY_DEB} || true  # probably fails...
apt-get install -fy  # install dependencies and package

# install system-level extras
apt-get install -y \
    ${PY_PREFIX}-nds2-client \
    ${PY_PREFIX}-h5py \
|| true
