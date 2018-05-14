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

# enable lscsoft jessie-proposed (first required for python-tqdm gwpy/gwpy#735)
if [[ `cat /etc/debian_version` == "8."* ]]; then
    cat << EOF > /etc/apt/sources.list.d/lscsoft-proposed.list
deb http://software.ligo.org/lscsoft/debian jessie-proposed contrib
EOF
    cat << EOF > /etc/apt/preferences.d/lscsoft-proposed.pref
Package: *
Pin: release n=jessie-proposed
Pin-Priority: 100
EOF
    apt-get update -yqq
fi

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

# get versions
GWPY_VERSION=`python setup.py version | grep Version | cut -d\  -f2`
GWPY_RELEASE=${GWPY_VERSION%%+*}

# upgrade setuptools for development builds only to prevent version munging
if [[ "${GWPY_VERSION}" == *"+"* ]]; then
    pip install "setuptools>=25"
fi

# upgrade GitPython (required for git>=2.15.0)
#     since we link the git clone from travis, the dependency is actually
#     fixed to the version of git on the travis image
pip install "GitPython>=2.1.8"

# prepare the tarball (sdist generates debian/changelog)
python setup.py sdist

# make the debian package
mkdir -p dist/debian
pushd dist/debian
cp ../gwpy-${GWPY_VERSION}.tar.gz ../gwpy_${GWPY_RELEASE}.orig.tar.gz
tar -xf ../gwpy_${GWPY_RELEASE}.orig.tar.gz --strip-components=1
dpkg-buildpackage -us -uc
popd

if [ ${PY_XY} -lt 30 ]; then  # install python2 only
    PREFICES="python"
else  # install both 2 and 3
    PREFICES="python python3"
fi
for PREF in ${PREFICES}; do
    # print and install the deb
    GWPY_DEB="dist/${PREF}-gwpy_${GWPY_RELEASE}-1_all.deb"
    echo "-------------------------------------------------------"
    dpkg --info ${GWPY_DEB}
    echo "-------------------------------------------------------"
    dpkg --install ${GWPY_DEB} || { \
        apt-get -y -f install;  # install dependencies and package
        apt-get -yq install ${PREF}-tqdm;  # install tqdm from backports
        dpkg --install ${GWPY_DEB};  # shouldn't fail
    }
done

# install system-level extras for the correct python version
for pckg in \
    libroot-bindings-python5.34 libroot-tree-treeplayer-dev \
    libroot-math-physics-dev libroot-graf2d-postscript-dev \
    ${PY_PREFIX}-nds2-client \
    ${PY_PREFIX}-dqsegdb ${PY_PREFIX}-m2crypto \
    ${PY_PREFIX}-sqlalchemy \
    ${PY_PREFIX}-pandas \
    ${PY_PREFIX}-psycopg2 \
    ${PY_PREFIX}-pymysql \
    ldas-tools-framecpp-${PY_PREFIX} \
    lalframe-${PY_PREFIX} \
    lalsimulation-${PY_PREFIX} \
    ${PY_PREFIX}-h5py \
; do
    apt-get -yqq install $pckg || true
done
