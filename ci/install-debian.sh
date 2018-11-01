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

# -- setup --------------------------------------

if [[ "${PYTHON_VERSION}" = "2.7" ]]; then
    PY_PREFIX="python"
else
    PY_PREFIX="python3"
fi

# -- build --------------------------------------

mkdir build
pushd build

apt-get -yqq update

# install basic build dependencies
apt-get -yqq install dpkg-dev devscripts

# unwrap tarball into build path
tar -xf ../*.tar.* --strip-components=1

# install build requires
mk-build-deps --tool "apt-get -y" --install --remove

# build debian packages
dpkg-buildpackage -us -uc -b

popd

# -- install ------------------------------------

# print and install the deb
GWPY_DEB="${PY_PREFIX}-gwpy_*_all.deb"
echo "-------------------------------------------------------"
dpkg --info ${GWPY_DEB}
echo "-------------------------------------------------------"
dpkg --install ${GWPY_DEB} || { \
    apt-get -y -f install;  # install dependencies and package
    dpkg --install ${GWPY_DEB};  # shouldn't fail
}

# install extras
# NOTE: git is needed for coveralls
apt-get -yqq install \
    git \
    ${PY_PREFIX}-pip \
    libkrb5-dev krb5-user \
    dvipng texlive-latex-base texlive-latex-extra \
    ${PY_PREFIX}-glue \
    ${PY_PREFIX}-sqlalchemy \
    ${PY_PREFIX}-psycopg2 \
    ${PY_PREFIX}-pandas \
    ${PY_PREFIX}-pytest \
    ${PY_PREFIX}-pytest-cov \
    ${PY_PREFIX}-freezegun \
    ${PY_PREFIX}-sqlparse \
    ${PY_PREFIX}-bs4 \
    lal-${PY_PREFIX} \
    lalframe-${PY_PREFIX} \
    lalsimulation-${PY_PREFIX}

# install ROOT for python2 only
if [ "${PY_PREFIX}" == "python" ]; then
    apt-get -yqq install \
        libroot-bindings-python5.34 \
        libroot-tree-treeplayer-dev \
        libroot-math-physics-dev \
        libroot-graf2d-postscript-dev
fi

# HACK: fix missing file from ldas-tools-framecpp
if [ -d /usr/lib/${PYTHON}/dist-packages/LDAStools/ -a \
     ! -f /usr/lib/${PYTHON}/dist-packages/LDAStools/__init__.py ]; then
    touch /usr/lib/${PYTHON}/dist-packages/LDAStools/__init__.py
fi
