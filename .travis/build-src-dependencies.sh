#!/bin/bash

# set paths for PKG_CONFIG
export PKG_CONFIG_PATH=${PKG_CONFIG_PATH}:${VIRTUAL_ENV}/lib/pkgconfig

# use g++ 4.7 (mainly for framecpp)
export CXX=`which g++-4.7`
echo "Set CXX=${CXX}"

# get latest version of packages
. .travis/find-latest-releases.sh

# -- LAL ----------------------------------------------------------------------

bash -e .travis/build-with-autotools.sh \
    python-${TRAVIS_PYTHON_VERSION}-lal ${LAL} --enable-swig-python

# stop here if doing a minimal build
[[ "$MINIMAL" = true ]] && return

FAILURES=""

# -- framecpp -----------------------------------------------------------------

bash .travis/build-with-autotools.sh \
    python-${TRAVIS_PYTHON_VERSION}-ldas-tools-al \
    ${LDAS_TOOLS_AL} --enable-python || FAILURES="$FAILURES ldas-tools-al"

bash .travis/build-with-autotools.sh \
    python-${TRAVIS_PYTHON_VERSION}-framecpp \
    ${FRAMECPP} --enable-python || FAILURES="$FAILURES framecpp"

# -- lalframe -----------------------------------------------------------------

bash .travis/build-with-autotools.sh \
    python-${TRAVIS_PYTHON_VERSION}-lalframe \
    ${LALFRAME} --enable-swig-python || FAILURES="$FAILURES lalframe"

# -- NDS2 ---------------------------------------------------------------------

bash .travis/build-with-autotools.sh \
    python-${TRAVIS_PYTHON_VERSION}-nds2-client \
    ${NDS2_CLIENT} --disable-swig-java --disable-mex-matlab || FAILURES="$FAILURES nds2-client"

# -- exit ---------------------------------------------------------------------

if [[ -n "${FAILURES+x}" ]]; then
    echo "The following builds failed: ${FAILURES}"
fi
