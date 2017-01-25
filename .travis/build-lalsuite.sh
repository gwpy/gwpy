#!/bin/bash
#
# travis-ci build script for LALSuite packages that won't install using apt
# on ubuntu

# -- setup

pip install --quiet bs4 six
find_latest_version() {
    python .travis/find-latest-release.py -o version $@
}

# set paths for PKG_CONFIG
export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${VIRTUAL_ENV}/lib/pkgconfig

# find latest release version
LALSUITE_URL="http://software.ligo.org/lscsoft/source/lalsuite"
LAL_VERSION=`find_latest_version ${LALSUITE_URL} lal`
LAL="${LALSUITE_URL}/lal-${LAL_VERSION}.tar.gz"
LALFRAME_VERSION=`find_latest_version ${LALSUITE_URL} lalframe`
LALFRAME="${LALSUITE_URL}/lalframe-${LALFRAME_VERSION}.tar.gz"

# -- install

# build LAL
bash -e .travis/build-with-autotools.sh \
    lal-${LAL_VERSION}/${TRAVIS_PYTHON_VERSION} ${LAL} --enable-swig-python

# build LALFRAME
bash -e .travis/build-with-autotools.sh \
    lalframe-${LALFRAME_VERSION}/${TRAVIS_PYTHON_VERSION} ${LALFRAME} --enable-swig-python
