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
        bash -ec "$@"
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

get_os_version() {
    case "$(uname -s)" in
        Darwin)
            sw_vers -productVersion
            ;;
        *)
            source /etc/os-release
            echo ${VERSION_ID}
    esac
}

get_package_manager() {
    case "$(get_os_type)" in
        macos)
            echo port
            ;;
        centos|rhel|fedora)
            echo yum
            ;;
        *)
            echo apt-get
    esac
}

update_package_manager() {
    local pkger=`get_package_manager`
    case "$(get_package_manager)" in
        "port")
            port selfupdate
            ;;
        "apt-get")
            apt-get --yes --quiet update
            ;;
        "yum")
            yum clean all
            yum makecache
            yum -y update
            ;;
    esac
}

install_package() {
    local pkger=`get_package_manager`
    case "$pkger" in
        "port")
            port -N install $@
            ;;
        "apt-get")
            apt-get --yes --quiet install $@
            ;;
        "yum")
            yum -y install $@
            ;;
    esac
}

get_python_version() {
    if [ -z "${PYTHON_VERSION}" ]; then
        PYTHON_VERSION=$(python -c 'import sys; print(sys.version[:3])))')
    fi
    export PYTHON_VERSION
    echo ${PYTHON_VERSION}
}

get_python2_version() {
    echo '2.7'
}

get_python3_version() {
    case "$(get_os_type)$(get_os_version)" in
        centos7|debian8)
           echo '3.4'
           ;;
        debian9)
           echo '3.5'
           ;;
        debian10|macos*)
           echo '3.6'
           ;;
        macos*)
           echo '3.6'
           ;;
    esac
}

get_environment() {
    local pkger=$(get_package_manager)

    # OS variables
    export OS_NAME=$(get_os_type)
    export OS_VERSION=$(get_os_version)

    # PYTHON VARIABLES
    export PYTHON_VERSION=$(get_python_version)
    export PYTHON="python${PYTHON_VERSION}"
    export PYTHON2_VERSION=$(get_python2_version)
    export PYTHON2="python${PYTHON2_VERSION}"
    export PYTHON3_VERSION=$(get_python3_version)
    export PYTHON3="python${PYTHON3_VERSION}"

    IFS='.' read PY_MAJOR_VERSION PY_MINOR_VERSION <<< "${PYTHON_VERSION}"
    export PY_XY="${PY_MAJOR_VERSION}${PY_MINOR_VERSION}"

    export PIP="${PYTHON} -m pip"

    case "$pkger" in
        "port")
            PY_DIST="python${PY_XY}"
            PY_PREFIX="py${PY_XY}"
            PIP="sudo ${PIP}"
            ;;
        "apt-get")
            if [ ${PY_MAJOR_VERSION} == 2 ]; then
                PY_DIST="python"
                PY_PREFIX="python"
            else
                PY_DIST="python${PY_MAJOR_VERSION}"
                PY_PREFIX="python${PY_MAJOR_VERSION}"
            fi
            ;;
        "yum")
            if [ ${PY_MAJOR_VERSION} == 2 ]; then
                PY_DIST="python"
                PY_PREFIX="python"
            elif [ ${PY_XY} -eq 34 ]; then
                PY_DIST="python${PY_XY}"
                PY_PREFIX="python${PY_XY}"
            else
                PY_DIST="python${PY_XY}u"
                PY_PREFIX="python${PY_XY}u"
            fi
            ;;
    esac
    export PY_DIST PY_PREFIX PIP
}

install_python() {
    get_environment  # <- set python variables
    install_package ${PY_DIST} ${PY_PREFIX}-pip ${PY_PREFIX}-setuptools
}

# -- utilities ----------------------------------------------------------------

function write_visual_bells() {
  while :; do
    echo -en "\a"
    sleep 10
  done
}
