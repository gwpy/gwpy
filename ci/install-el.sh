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

. ci/lib.sh
get_environment

# install build dependencies
yum -y -q update
yum -y -q install rpm-build

# build package
rpmbuild --define "_rpmdir $(pwd)" -tb gwpy-*.tar.*
mv noarch/*.rpm .

# install the rpm
if [ ${PY_XY} -lt 30 ]; then
    GWPY_RPM="python2-gwpy-*.noarch.rpm"  # install python2 only
else
    GWPY_RPM="${PY_PREFIX}-gwpy-*.noarch.rpm"  # install both 2 and 3
fi
yum -y -q --nogpgcheck localinstall ${GWPY_RPM}

# install system-level extras that use python- prefix
yum -y -q install \
    nds2-client-${PY_PREFIX} \
    ldas-tools-framecpp-${PY_PREFIX} \
    lalframe-${PY_PREFIX} \
    lalsimulation-${PY_PREFIX} \
|| true  # don't fail on missing packages

# install system-level extras that might use python2- prefix
if [ ${PY_XY} -lt 30 ]; then
    yum -y -q install \
        python2-root \
        python2-freezegun \
        python2-pytest-cov
else
    yum -y -q install \
        ${PY_PREFIX}-pytest-cov \
        ${PY_PREFIX}-freezegun \
        ${PY_PREFIX}-root
fi

# HACK: fix missing file from ldas-tools-framecpp
if [ -d /usr/lib64/${PYTHON}/site-packages/LDAStools -a \
     ! -f /usr/lib64/${PYTHON}/site-packages/LDAStools/__init__.py ]; then
    touch /usr/lib64/${PYTHON}/site-packages/LDAStools/__init__.py
fi
