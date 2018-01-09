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

# install pip for system python
apt-get -yq install python-pip

# install build dependencies (should match debian/control)
apt-get -yq install \
    debhelper \
    dh-python \
    python-all \
    python3-all \
    python-setuptools \
    python3-setuptools \
    python-git \
    python-jinja2 \

# install setuptools from jessie-backports
if [ `get_debian_version` -eq 8 ]; then
    # enable backports
    apt-cache policy | grep "jessie-backports/main" &> /dev/null || \
        {
         echo "deb http://ftp.debian.org/debian jessie-backports main" \
         > /etc/apt/sources.list.d/backports.list;
         apt-get update -yqq;
        }
    # install setuptools
    apt-get -yq install -t jessie-backports \
        python-setuptools \
        python3-setuptools
fi

# get versions
GWPY_VERSION=`python setup.py version | grep Version | cut -d\  -f2`
GWPY_RELEASE=${GWPY_VERSION%%+*}

# prepare the tarball (sdist generates debian/changelog)
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
    apt-get -y -f install;  # install dependencies and package
    dpkg --install ${GWPY_DEB};  # shouldn't fail
}

# install system-level extras for the correct python version
for pckg in \
    libroot-bindings-python5.34 \
    ${PY_PREFIX}-nds2-client \
    ldas-tools-framecpp-${PY_PREFIX} \
    lalframe-${PY_PREFIX} \
    ${PY_PREFIX}-h5py \
; do
    apt-get -yqq install $pckg || true
done

if [ ${PY_XY} -lt 30 ]; then
    NO_ROOT_NUMPY_TMVA=1 ${PIP} install root_numpy ${PIP_FLAGS}
fi
