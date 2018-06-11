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

get_os_type() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo $ID
    elif [[ ${TRAVIS_OS_NAME} == "osx" ]] || [[ "$(uname)" == "Darwin" ]]; then
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

get_python_version() {
    if [ -z "${PYTHON_VERSION}" ]; then
        PYTHON_VERSION=$(python -c 'import sys; print(sys.version[:3])')
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
        *)
           echo '3.6'
           ;;
    esac
}

get_environment() {
    # OS variables
    export OS_NAME=$(get_os_type)

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

    case "${OS_NAME}${PY_MAJOR_VERSION}" in
        macos*)  # macports
            PY_DIST="python${PY_XY}"
            PY_PREFIX="py${PY_XY}"
            PIP="sudo ${PIP}"
            ;;
        debian2)
            PY_DIST="python"
            PY_PREFIX="python"
            ;;
        debian3)
            PY_DIST="python${PY_MAJOR_VERSION}"
            PY_PREFIX="python${PY_MAJOR_VERSION}"
            ;;
        centos2|rhel2|fedora2)
            PY_DIST="python"
            PY_PREFIX="python"
            ;;
        centos3|rhel3|fedora2)
            if [ "${PYTHON_VERSION}" == "${PYTHON3_VERSION}" ]; then # IUS
                PY_DIST="python${PY_XY}u"
                PY_PREFIX="python${PY_XY}u"
            else  # base repo
                PY_DIST="python${PY_XY}"
                PY_PREFIX="python${PY_XY}"
            fi
            ;;
    esac
    export PY_DIST PY_PREFIX PIP
}

# -- utilities ----------------------------------------------------------------

function write_visual_bells() {
  while :; do
    echo -en "\a"
    sleep 10
  done
}
