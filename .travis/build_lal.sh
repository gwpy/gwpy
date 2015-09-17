#!/bin/bash
set -e

LSCSOFT_LOCATION=${HOME}/lscsoft
mkdir -p ${LSCSOFT_LOCATION}

LSCSOFT_SOURCE_URL="http://software.ligo.org/lscsoft/source"

[[ -z ${PYTHON_VERSION} ]] && PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

build_lscsoft_package() {
    local package=$1
    local path=$2
    if [ -d ${LSCSOFT_LOCATION}/${package} ]; then
        return 0
    else
        # install
        rm -f ${package}.tar.gz
        if [ -n "$path" ]; then
            wget --quiet ${LSCSOFT_SOURCE_URL}/$path/${package}.tar.gz
        else
            wget --quiet ${LSCSOFT_SOURCE_URL}/${package}.tar.gz
        fi
        tar -zxf ${package}.tar.gz
        cd ${package}
        ./configure --quiet --enable-silent-rules --prefix ${LSCSOFT_LOCATION}/${package} && make && make install && cd -
    fi
}

get_python_path() {
    local package=$1
    echo ${LSCSOFT_LOCATION}/${package}/lib/python${PYTHON_VERSION}/site-packages
}

get_pkgconfig_path() {
    local package=$1
    echo ${LSCSOFT_LOCATION}/${package}/lib/pkgconfig
}

# list packages
PACKAGES="
libframe-8.20
ldas-tools-2.4.1
lal-6.15.0
lalframe-1.3.0
"

# build all packages
for _package in ${PACKAGES}; do
    [[ ${_package} == "lal"* ]] && _path="lalsuite" || _path=""
    build_lscsoft_package ${_package} ${_path}
    if [ -d `get_pkgconfig_path ${_package}` ]; then
        export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:`get_pkgconfig_path ${_package}`
    fi
    PYTHONPATH=${PYTHONPATH}:`get_python_path ${_package}`
done

export PYTHONPATH
