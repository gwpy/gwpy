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

set -e
set -x

#
# Configure the relevant docker container for this CI job, then install
# the dependencies as system packages
#

if [ -z ${DOCKER_IMAGE} ]; then
    echo "Not a docker install, skipping docker configuration"
else
    sudo apt-get update -yqq
    sudo docker pull ${DOCKER_IMAGE}
    sudo docker run \
        --rm=true \
        --detach \
        --name ${GWPY_CI} \
        --volume `pwd`:/gwpy:rw \
        ${DOCKER_IMAGE}
    sleep 10
fi
