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

yum clean all
yum makecache
yum -y update

yum -y install \
    rpm-build \
    ${PYPKG_PREFIX} \
    ${PYPKG_PREFIX}-pip \
    ${PYPKG_PREFIX}-virtualenv

# create virtualenv for this build
ci_virtualenv

# build the RPM
python setup.py bdist_rpm \
    --fix-python \
    --changelog="`python changelog.py --start-tag 'v0.5'`"

# exit virtualenv
ci_clean_virtualenv

# install the rpm
rpm -ivh dist/gwpy-${GWPY_VERSION}-1.noarch.rpm

# install system-level extras
yum -y install \
    nds2-client-python \
    h5py \
|| true
