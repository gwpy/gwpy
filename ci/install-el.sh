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

# -- prep ---------------------------------------------------------------------

. ci/lib.sh
get_environment
PY3_PREFIX=${PYTHON3/./}

# install build dependencies (should match etc/spec.template)
yum -y -q update
yum -y -q install \
    rpm-build \
    python-rpm-macros \
    python2-rpm-macros \
    python3-rpm-macros \
    python2-setuptools \
    ${PY3_PREFIX}-setuptools

# -- build --------------------------------------------------------------------

rpmbuild --define "_rpmdir $(pwd)" -tb gwpy-*.tar.*
mv noarch/*.rpm .

# install the rpm
if [ ${PY_XY} -lt 30 ]; then
    GWPY_RPM="python2-gwpy-*.noarch.rpm"  # install python2 only
else
    GWPY_RPM="${PY_PREFIX}-gwpy-*.noarch.rpm"  # install both 2 and 3
fi
yum -y -q --nogpgcheck localinstall ${GWPY_RPM}

# -- extras -------------------------------------------------------------------

# install extras using modern python prefices
# NOTE: git is needed for coveralls
yum -y -q install \
    git \
    ${PY_PREFIX}-pip \
    ${PY_PREFIX}-root \
    ${PY_PREFIX}-freezegun \
    ${PY_PREFIX}-pytest-cov

# install LIGO extras for python2 only
if [ ${PY_XY} -lt 30 ]; then
    yum -y install \
        nds2-client-python
        ldas-tools-framecpp-python \
        lalframe-python \
        lalsimulation-python
fi

# HACK: fix missing file from ldas-tools-framecpp
if [ -d /usr/lib64/${PYTHON}/site-packages/LDAStools -a \
     ! -f /usr/lib64/${PYTHON}/site-packages/LDAStools/__init__.py ]; then
    touch /usr/lib64/${PYTHON}/site-packages/LDAStools/__init__.py
fi
