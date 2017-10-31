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
# Library functions for CI builds
#

# set environment
# NOTE: any variables defined here should also be added
#       to the `docker run` command in /ci/docker-install.sh
#       via --env options

if [ -z ${DOCKER_IMAGE} ]; then
    GWPY_PATH="`pwd`/"
    PIP="pip"
    PYTHON="python"
else
    GWPY_PATH="/gwpy/"
    PYTHON="python${PYTHON_VERSION}"
    if [[ "${PYTHON_VERSION}" != "2"* ]] && [[ "${DOCKER_IMAGE}" == *"el7" ]]; then
        PIP="pip${PYTHON_VERSION:0:1}"
        PYPKG_PREFIX="python${PYTHON_VERSION//./}"
    elif [[ "${PYTHON_VERSION}" != "2"* ]]; then
        PIP="pip${PYTHON_VERSION:0:1}"
        PYPKG_PREFIX="python${PYTHON_VERSION:0:1}"
    else
        PIP="pip"
        PYPKG_PREFIX="python"
    fi
fi


ci_run() {
    set -x
    if [ -z ${DOCKER_IMAGE} ]; then  # execute function normally
        eval "$@" || return 1
    else  # execute function in docker container
        docker exec -it ${DOCKER_IMAGE##*:} sh -lxec "cd ${GWPY_PATH}; eval \"$@\"" || return 1
    fi
    set +x
    return 0
}
