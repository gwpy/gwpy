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
        bash -lec "$@"
    else  # execute function in docker container
        docker exec -it ${DOCKER_IMAGE##*:} bash -lec "$@"
    fi
}

# -- in-container helpers -----------------------------------------------------

get_os_type() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo $ID
    elif [[ ${TRAVIS_OS_NAME} == "osx" ]] || [[ "`uname`" == "Darwin" ]]; then
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

update_package_manager() {
    local pkger=`get_package_manager`
    case "$pkger" in
        "port")
            port selfupdate
            ;;
        "apt-get")
            apt-get -yq update
            ;;
        "yum")
            yum clean all
            yum makecache
            yum -y update
            ;;
    esac
}

get_python_version() {
    if [ -n ${PYTHON_VERSION} ]; then
        :
    elif [ -n ${TRAVIS_PYTHON_VERSION} ]; then
        PYTHON_VERSION=${TRAVIS_PYTHON_VERSION}
    else
        PYTHON_VERSION=`python -c
            'import sys; print(".".join(map(str, sys.version_info[:2])))'`
    fi
    export PYTHON_VERSION
    echo ${PYTHON_VERSION}
}

get_environment() {
    local pkger=`get_package_manager`
    local pyversion=`get_python_version`
    IFS='.' read PY_MAJOR_VERSION PY_MINOR_VERSION <<< "$pyversion"
    PY_XY="${PY_MAJOR_VERSION}${PY_MINOR_VERSION}"
    PYTHON=python$pyversion
    case "$pkger" in
        "port")
            PY_DIST=python${PY_XY}
            PY_PREFIX=py${PY_XY}
            PIP=pip-$pyversion
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
                PIP=pip$pyversion
            fi
            ;;
    esac
    export PYTHON PY_MAJOR_VERSION PY_MINOR_VERSION PY_XY PY_DIST PY_PREFIX PIP
}

install_python() {
    local pkger=`get_package_manager`
    get_environment  # <- set python variables
    $pkger install \
        ${PY_DIST} \
        ${PY_PREFIX}-pip \
        ${PY_PREFIX}-setuptools
}

get_configparser_option() {
    local fp=$1
    local section=$2
    local option=$3
    $PYTHON -c "
from configparser import ConfigParser
cp = ConfigParser(defaults={'py-prefix': '${PY_PREFIX}'})
cp.read('$fp');
cp.set('$section', 'py-prefix', '${PY_PREFIX}')
print(cp.get('$section', '$option'))"
}
