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

# update system
yum clean all
yum makecache
yum -y update
yum -y install rpm-build git2u

if [ -z $PYTHON ]; then  # correct python version not installed
    yum -y install ${PY_DIST} ${PY_PREFIX}-devel
    PYTHON=`which python${PYTHON_VERSION}`
fi

# install pip
yum install -y ${PY_PREFIX}-pip

create_virtualenv

# build the RPM
GWPY_VERSION=`python setup.py version | grep Version | cut -d\  -f2`
python setup.py bdist_rpm \
    --python ${PYTHON} \
    --changelog="`python changelog.py --start-tag 'v0.5'`"

clean_virtualenv

# install the rpm
rpm -ivh dist/gwpy-${GWPY_VERSION}-1.noarch.rpm

# install system-level extras
yum -y install \
    nds2-client-python \
    h5py \
|| true
