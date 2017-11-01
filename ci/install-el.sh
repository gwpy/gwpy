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

set -x

yum update -y

yum install \
    ${PYPKG_PREFIX} \
    ${PYPKG_PREFIX}-pip

${PIP} install GitPython  # needed for changelog.py

# build the RPM
${PYTHON} setup.py bdist_rpm \
    --fix-python \
    --changelog="`${PYTHON} changelog.py --start-tag 'v0.5'`"

# install the rpm
rpm -ivh dist/gwpy-${GWPY_VERSION}-1.noarch.rpm

# install system-level extras
yum install \
    nds2-client-python \
    h5py \
|| true

set +x
