#!/bin/bash
# Copyright (C) Duncan Macleod (2017-2020)
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

# install basic build dependencies
yum -y -q update
yum -y -q install lscsoft-testing-config
yum -y -q update
yum -y -q install \
    rpm-build \
    yum-utils \
    epel-rpm-macros

# correct issue with missing tzdata files
# https://listserv.fnal.gov/scripts/wa.exe?A2=ind1910&L=SCIENTIFIC-LINUX-USERS&P=21164
yum -y -q reinstall tzdata

# determine python package prefix
if [ "${PYTHON_VERSION:0:1}" -eq 3 ]; then
    PY_PREFIX="python$(rpm --eval "%python3_pkgversion")"
else
    PY_PREFIX="python${PYTHON_VERSION:0:1}"
fi

# -- build --------------------------------------

TOPDIR=$(pwd)
TARBALL="$(pwd)/gwpy-*.tar.*"

mkdir build
pushd build

# build src rpm
SRPM=$(rpmbuild --define "_topdir ${TOPDIR}" -ts ${TARBALL} | cut -d\  -f2)

# install BuildRequires
yum-builddep -y -q ${SRPM}

# build binary rpm(s)
rpmbuild --define "_topdir ${TOPDIR}" --rebuild ${SRPM}

popd

# print info
rpm -qilp ${TOPDIR}/RPMS/*/*gwpy*.rpm
rpm -qp --provides ${TOPDIR}/RPMS/*/*gwpy*.rpm

# -- install ------------------------------------

RPM="${TOPDIR}/RPMS/noarch/${PY_PREFIX}-gwpy-*.rpm"
yum -y -q --nogpgcheck localinstall ${RPM}

# -- extras -------------------------------------

# install system-level extras
yum -y -q install \
    which \
    ${PY_PREFIX}-beautifulsoup4 \
    ${PY_PREFIX}-dqsegdb \
    ${PY_PREFIX}-freezegun \
    ${PY_PREFIX}-glue \
    ${PY_PREFIX}-lalframe \
    ${PY_PREFIX}-lalsimulation \
    ${PY_PREFIX}-ligo-lw \
    ${PY_PREFIX}-m2crypto \
    ${PY_PREFIX}-nds2-client \
    ${PY_PREFIX}-pandas \
    ${PY_PREFIX}-pip \
    ${PY_PREFIX}-psycopg2 \
    ${PY_PREFIX}-PyMySQL \
    ${PY_PREFIX}-pytest \
    ${PY_PREFIX}-pytest-cov \
    ${PY_PREFIX}-root \
    ${PY_PREFIX}-sqlparse \
    ${PY_PREFIX}-sqlalchemy \
    texlive-dvipng-bin texlive-latex-bin-bin \
    texlive-type1cm texlive-collection-fontsrecommended \
;
