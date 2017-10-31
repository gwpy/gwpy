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

[ -z ${TRAVIS_SECURE_ENV_VARS} ] && return 0

#
# Build system packages
#

GWPY_VERSION=`python setup.py version | grep Version | cut -d\  -f2`

if [[ "${DOCKER_IMAGE}" == *"jessie" ]]; then
    # prepare the tarball
    pip install stdeb
    python changelog.py --changelog-format=deb > debian/changelog
    python setup.py sdist

    # make the debian package
    mkdir -p dist/debian
    cd dist/debian
    cp ../gwpy-${GWPY_VERSION}.tar.gz ../gwpy_${GWPY_VERSION}.orig.tar.gz
    dpkg-buildpackage -us -uc

    # install the deb
    sudo dpkg -i ../python-gwpy_${GWPY_VERSION}-1_all.deb

elif [[ "${DOCKER_IMAGE}" == *"el7" ]]; then
    # build the RPM
    python setup.py bdist_rpm --changelog=`python changelog.py`

    # install the rpm
    sudo rpm -ivh dist/gwpy-${GWPY_VERSION}-1.noarch.rpm

fi
