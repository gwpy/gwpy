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
# Build RedHat (Enterprise Linux) packages
#

# -- setup --------------------------------------

# determine python prefix, even though we only install for python2

if [[ "${PYTHON_VERSION}" == "2.7" ]]; then
    PY_PREFIX="python2"
else
    PY_PREFIX="python${PYTHON_VERSION/./}"
fi

# -- build --------------------------------------

TOPDIR=$(pwd)
TARBALL="$(pwd)/gwpy-*.tar.*"

mkdir build
pushd build

yum -y -q update

# install basic build dependencies
yum -y -q install \
    rpm-build \
    yum-utils

# build src rpm
SRPM=$(rpmbuild --define "_topdir ${TOPDIR}" -ts ${TARBALL} | cut -d\  -f2)

# install BuildRequires
yum-builddep -y -q ${SRPM}

# build binary rpm(s)
rpmbuild --define "_topdir ${TOPDIR}" --rebuild ${SRPM}

popd

# -- install ------------------------------------

RPM="${TOPDIR}/RPMS/noarch/${PY_PREFIX}-gwpy-*.rpm"
yum -y -q --nogpgcheck localinstall ${RPM}

# -- extras -------------------------------------
#
# This is explicitly only set up for python2.7 on RHEL7
#

# install system-level extras
yum -y -q install \
    python2-pip \
    python2-pytest \
    python2-pytest-cov \
    python2-mock \
    python2-freezegun \
    python-sqlparse \
    python-beautifulsoup4 \
    python-sqlalchemy \
    python2-PyMySQL \
    python2-gwdatafind \
    m2crypto \
    glue \
    dqsegdb \
    python-psycopg2 \
    python-pandas \
    python2-root \
    python2-nds2-client \
    python2-ldas-tools-framecpp \
    python2-lalframe \
    python2-lalsimulation \
    texlive-dvipng-bin texlive-latex-bin-bin \
    texlive-type1cm texlive-collection-fontsrecommended

# HACK: fix missing file from ldas-tools-framecpp
if [ -d /usr/lib64/${PYTHON}/site-packages/LDAStools -a \
     ! -f /usr/lib64/${PYTHON}/site-packages/LDAStools/__init__.py ]; then
    touch /usr/lib64/${PYTHON}/site-packages/LDAStools/__init__.py
fi
