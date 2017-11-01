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

GWPY_RELEASE=${GWPY_VERSION%%+*}
GWPY_VERSION=${GWPY_VERSION/+/-}

# install build dependencies
apt-get update -yqq
apt-get install -yqq \
    debhelper \
    ${PYPKG_PREFIX} \
    ${PYPKG_PREFIX}-all \
    ${PYPKG_PREFIX}-pip \
    lal-${PYPKG_PREFIX}

# install build helpers
${PIP} install GitPython

# prepare the tarball
${PYTHON} changelog.py -f deb -s "v0.5" > debian/changelog
${PYTHON} setup.py sdist

# make the debian package
mkdir -p dist/debian
pushd dist/debian
cp ../gwpy-${GWPY_VERSION}.tar.gz ../gwpy_${GWPY_VERSION}.orig.tar.gz
tar -xf ../gwpy_${GWPY_VERSION}.orig.tar.gz --strip-components=1
dpkg-buildpackage -us -uc
popd

# print and install the deb
GWPY_DEB="dist/${PYPKG_PREFIX}-gwpy_${GWPY_RELEASE}-1_all.deb"
echo "-------------------------------------------------------"
dpkg --info ${GWPY_DEB}
echo "-------------------------------------------------------"
dpkg --install ${GWPY_DEB} || true  # probably fails...
apt-get -fy install  # install dependencies and package

# install system-level extras
apt-get install -y \
    ${PYPKG_PREFIX}-nds2-client \
    ${PYPKG_PREFIX}-h5py \
|| true
