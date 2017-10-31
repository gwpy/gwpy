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
# Build system packages
#

cd ${GWPY_PATH}
GWPY_VERSION=`${PYTHON} setup.py version | grep Version | cut -d\  -f2`

if [ -z ${DOCKER_IMAGE} ]; then  # simple
    ${PIP} install .

elif [[ "${DOCKER_IMAGE}" == *"jessie" ]]; then  # Debian

    GWPY_VERSION=${GWPY_VERSION/+/-}

    apt-get install -yqq \
        ${PYPKG_PREFIX} \
        ${PYPKG_PREFIX}-pip

    ${PIP} install stdeb GitPython

    # prepare the tarball
    ${PYTHON} changelog.py --start-tag "v0.5" > debian/changelog
    ${PYTHON} setup.py sdist

    # make the debian package
    mkdir -p dist/debian
    pushd dist/debian
    cp ../gwpy-${GWPY_VERSION}.tar.gz ../gwpy_${GWPY_VERSION}.orig.tar.gz
    tar -xf ../gwpy_${GWPY_VERSION}.orig.tar.gz --strip-components=1
    dpkg-buildpackage -us -uc
    popd

    # install the deb
    dpkg -i dist/python-gwpy_${GWPY_VERSION}-1_all.deb

    # install system-level extras
    apt-get install -y \
        ${PYPKG_PREFIX}-nds2-client \
        ${PYPKG_PREFIX}-h5py \
    || true

elif [[ "${DOCKER_IMAGE}" == *"el7" ]]; then  # Scientific Linux

    yum install \
        ${PYPKG_PREFIX} \
        ${PYPKG_PREFIX}-pip

    pip install GitPython  # needed for changelog.py

    # build the RPM
    ${PYTHON} setup.py bdist_rpm --fix-python --changelog="`${PYTHON} changelog.py --start-tag 'v0.5'`"

    # install the rpm
    rpm -ivh dist/gwpy-${GWPY_VERSION}-1.noarch.rpm

    # install system-level extras
    yum install \
        nds2-client-python \
        h5py \
    || true

fi

# install extras
${PIP} install -r requirements-dev.txt

echo "------------------------------------------------------------------------"
echo
echo "GWpy installed to `${PYTHON} -c 'import gwpy; print(gwpy.__file__)'`"
echo
echo "------------------------------------------------------------------------"
