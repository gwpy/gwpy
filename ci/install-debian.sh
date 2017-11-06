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

# install build dependencies
apt-get update -yqq
apt-get install -yqq \
    debhelper \
    dh-python \
    python-all \
    python-setuptools \
    python-pip \
    python-git \
    ${PY_PREFIX}-setuptools

# needed to prevent version number munging with versioneer
pip install "setuptools>33"

# get versions
GWPY_VERSION=`python setup.py version | grep Version | cut -d\  -f2`
GWPY_RELEASE=${GWPY_VERSION%%+*}

# prepare the tarball
python changelog.py -f deb -s "v0.5" > debian/changelog
python setup.py sdist

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
dpkg --install ${GWPY_DEB} || { \
    apt-get install -fy;  # install dependencies and package
    dpkg --install ${GWPY_DEB};  # shouldn't fail
}

# install system-level extras for the correct python version
apt-get install -y --ignore-missing \
    ${PY_PREFIX}-nds2-client \
    ldas-tools-framecpp-${PY_PREFIX} \
    lalframe-${PY_PREFIX} \
    ${PY_PREFIX}-h5py
