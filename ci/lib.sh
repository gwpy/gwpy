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
# Can be used outside and inside docker containers
#

# set path to build directory
if [ -z "${DOCKER_IMAGE}" ]; then
    GWPY_PATH=`pwd`
else
    GWPY_PATH="/gwpy"
fi
export GWPY_PATH

# -- out-of-container helpers -------------------------------------------------

ci_run() {
    # run a command normally, or in docker, depending on environment
    if [ -z "${DOCKER_IMAGE}" ]; then  # execute function normally
        eval "$@" || return 1
    else  # execute function in docker container
        docker exec -it ${DOCKER_IMAGE##*:} bash -lc "eval \"$@\"" || return 1
    fi
    return 0
}

# -- in-container helpers -----------------------------------------------------

get_os_type() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo $ID
    elif [ ${TRAVIS_OS_NAME} == "osx" ]; then
        echo macos
    fi
}

get_package_manager() {
    local ostype=`get_os_type`
    if [ $ostype == macos ]; then
        echo port
    elif [[ $ostype =~ ^(centos|rhel|fedora)$ ]]; then
        echo yum
    else
        echo apt-get
    fi
}

get_environment() {
    [ -z ${PYTHON_VERSION} ] && PYTHON_VERSION=`
        python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))'`
    local pkger=`get_package_manager`
    IFS='.' read PY_MAJOR_VERSION PY_MINOR_VERSION <<< "$PYTHON_VERSION"
    PY_XY="${PY_MAJOR_VERSION}${PY_MINOR_VERSION}"
    PYTHON=`which python${PYTHON_VERSION} || echo ""`
    case "$pkger" in
        "port")
            PY_DIST=python${PY_XY}
            PY_PREFIX=py${PY_XY}
            PIP=pip-${PYTHON_VERSION}
            ;;
        "apt-get")
            if [ ${PY_MAJOR_VERSION} == 2 ]; then
                PY_DIST=python
                PY_PREFIX=python
                PIP=pip
            else
                PY_DIST=python${PY_MAJOR_VERSION}
                PY_PREFIX=python${PY_MAJOR_VERSION}
                PIP=pip${PY_MAJOR_VERSION}
            fi
            ;;
        "yum")
            if [ ${PY_MAJOR_VERSION} == 2 ]; then
                PY_DIST=python
                PY_PREFIX=python
                PIP=pip
            elif [ ${PY_XY} -eq 34 ]; then
                PY_DIST=python${PY_XY}
                PY_PREFIX=python${PY_XY}
                PIP=pip${PY_MAJOR_VERSION}
            else
                PY_DIST=python${PY_XY}u
                PY_PREFIX=python${PY_XY}u
                PIP=pip${PY_MAJOR_VERSION}
            fi
            ;;
    esac
    export PYTHON PY_MAJOR_VERSION PY_MINOR_VERSION PY_XY PY_DIST PY_PREFIX PIP
}

create_virtualenv() {
    get_environment

    local pkger=`get_package_manager`
    $pkger install \
        $PY_DIST \
        python-virtualenv

    # create virtualenv in which to build
    virtualenv -p python${PYTHON_VERSION} ${GWPY_PATH}/opt/buildenv --clear --system-site-packages
    . ${GWPY_PATH}/opt/buildenv/bin/activate
    pip install --upgrade pip
    pip install --upgrade setuptools GitPython
}

clean_virtualenv() {
    local VENV_DIR=${VIRTUAL_ENV}
    deactivate
    rm -rf ${VENV_DIR}
}
