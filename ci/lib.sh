#!/bin/bash
#
# common functions for continuous integration in bash for GWpy
#
# Copyright (C) Cardiff University (2020)
#
# Author: Duncan Macleod <duncan.macleod@ligo.org>
#

# initialisation
export GWPY_CONDA_ENV_NAME="gwpyci"
export PYTHON_VERSION=$(echo "${PYTHON_VERSION:-${TRAVIS_PYTHON_VERSION}}" | cut -d. -f-2)

# functions

function on_windows() {
    if [[ "$(uname)" == "MSYS"* ]] || [[ "$(uname)" == "MINGW"* ]]; then
        return 0
    fi
    return 1
}

# --
# get path to python and pip
# (on Windows you don't get pythonX or pythonX.Y executables)
# --
function find_python_and_pip() {
    conda_activate
    if on_windows; then
        export PYTHON="python"
    else
        export PYTHON=$(which "python${PYTHON_VERSION}")
    fi
    export PIP="${PYTHON} -m pip"
}

# --
# install miniconda
# --
function install_miniconda() {
    if ! find_conda 1> /dev/null; then
        CONDA_ROOT=${CONDA_ROOT:-${HOME}/miniconda}
        if test ! -f ${CONDA_ROOT}/etc/profile.d/conda.sh; then
            # install conda
            [ "$(uname)" == "Darwin" ] && MC_OSNAME="MacOSX" || MC_OSNAME="Linux"
            MINICONDA="Miniconda${PYTHON_VERSION%%.*}-latest-${MC_OSNAME}-x86_64.sh"
            curl -L https://repo.continuum.io/miniconda/${MINICONDA} -o miniconda.sh
            bash miniconda.sh -b -u -p ${CONDA_ROOT}
        fi
    else
        CONDA_ROOT=$(conda info --base)
    fi
    source ${CONDA_ROOT}/etc/profile.d/conda.sh
    hash -r
}

# --
# find conda on the PATH
# --
function find_conda() {
    if [[ ! -z ${CONDA_EXE} ]]; then
        echo ${CONDA_EXE}
    else
        which conda
    fi
}

# --
# initialise conda in this bash session
# --
function conda_init() {
    if [[ "${USE_CONDA}"  == "false" ]]; then return 0; fi
    CONDA_ROOT="${CONDA_ROOT:-$(${CONDA_EXE:=conda} info --base)}"
    source ${CONDA_ROOT}/etc/profile.d/conda.sh
}

# --
# activate the CI conda environment
# --
function conda_activate() {
    if [[ "${USE_CONDA}"  == "false" ]]; then return 0; fi
    conda_init
    conda activate "${GWPY_CONDA_ENV_NAME}"
}
