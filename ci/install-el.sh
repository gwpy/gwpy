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

# update system
yum -y -q update

# install sdist dependencies
yum -y -q install \
    git \
    python2-pip \
    python-jinja2 \
    GitPython

# install build dependencies
yum -y -q install \
    rpm-build \
    python-rpm-macros \
    python2-rpm-macros \
    python2-setuptools

GWPY_VERSION=$(python setup.py --version)

# upgrade setuptools for development builds only to prevent version munging
if [[ "${GWPY_VERSION}" == *"+"* ]]; then
    pip install "setuptools>=25"
fi

# -- build and install --------------------------------------------------------

# build the RPM using tarball
python setup.py --quiet sdist
rpmbuild --define "_rpmdir $(pwd)/dist" -tb dist/gwpy-*.tar.gz

# install the rpm
GWPY_RPM="dist/noarch/python2-gwpy-*.noarch.rpm"
yum -y -q --nogpgcheck localinstall ${GWPY_RPM}

# -- third-party packages -----------------------------------------------------

# install system-level extras
yum -y -q install \
    python2-pip \
    python2-pytest \
    python-coverage \
    python2-mock \
    python2-freezegun \
    python-sqlparse \
    python-beautifulsoup4 \
    python-sqlalchemy \
    python2-PyMySQL \
    python-m2crypto \
    glue \
    dqsegdb \
    python-psycopg2 \
    python-pandas \
    python2-root \
    nds2-client-python \
    ldas-tools-framecpp-python \
    lalframe-python \
    lalsimulation-python

# HACK: fix missing file from ldas-tools-framecpp
if [ -d /usr/lib64/$PYTHON/site-packages/LDAStools -a \
     ! -f /usr/lib64/$PYTHON/site-packages/LDAStools/__init__.py ]; then
    touch /usr/lib64/$PYTHON/site-packages/LDAStools/__init__.py
fi
