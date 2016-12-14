#!/bin/bash

# install LAL first (always required)
bash -e .travis/build-lal.sh

# stop here if doing a minimal build
[[ "$MINIMAL" = true ]] && return

# -- install everything else

# set paths for PKG_CONFIG
export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${VIRTUAL_ENV}/lib/pkgconfig

FAILURES=""

# frameCPP
bash .travis/build-with-autotools.sh ldas-tools-al-${LDAS_TOOLS_AL_VERSION}/${TRAVIS_PYTHON_VERSION} ${LDAS_TOOLS_AL} || FAILURES="$FAILURES ldas-tools-al"
bash .travis/build-with-autotools.sh framecpp-${FRAMECPP_VERSION}/${TRAVIS_PYTHON_VERSION} ${FRAMECPP} --enable-python || FAILURES="$FAILURES framecpp"

# libframe
bash .travis/build-with-autotools.sh libframe-${LIBFRAME_VERSION}/${TRAVIS_PYTHON_VERSION} ${LIBFRAME} || FAILURES="$FAILURES libframe"

# LALFrame
bash .travis/build-with-autotools.sh lalframe-${LALFRAME_VERSION}/${TRAVIS_PYTHON_VERSION} ${LALFRAME} --enable-swig-python || FAILURES="$FAILURES lalframe"

# NDS2 client
bash .travis/build-with-autotools.sh nds2-client-${NDS2_CLIENT_VERSION}/${TRAVIS_PYTHON_VERSION} ${NDS2_CLIENT} --disable-swig-java --disable-mex-matlab || FAILURES="$FAILURES nds2-client"

if [[ -n "${FAILURES+x}" ]]; then
    echo "The following builds failed: ${FAILURES}"
fi
